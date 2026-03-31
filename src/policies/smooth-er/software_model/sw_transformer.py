from software_model.sw_operators import(
    Operator,
    Reshape,
    Concat,
    Transpose,
)

from software_model.sw_matmul import Matmul, BatchedMatmul
from software_model.sw_softmax import Softmax
from software_model.sw_layernorm import LayerNorm
from software_model.sw_gelu import GeLU
from software_model.sw_flashattention import FlashAttention

from software_model.sw_utils import Tensor, DataType
from software_model.sw_communication_primitives import AllReduceMultiPCB
from hardware_model.hw_system import System

import math
class TransformerBlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, system: System, config_file, use_flash_attention=False):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model #hidden dimension
        self.n_heads = n_heads
        self.device_count = device_count
        self.use_flash_attention = use_flash_attention
        self.device = system.device
        self.config_file = config_file

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)

        ## Flash attention operators
#        if self.use_flash_attention:
#            self.flash_attention_blocks = []
#            for i in range(self.n_heads):
#                self.flash_attention_blocks.append(self.init_flash_attention_block(data_type))
        self.MHA_reshape = Reshape(data_type)

        ## multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.flash_attention = FlashAttention(data_type)

        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    
    def __call__(self, x: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        # M: sram size skkim

        b, s, d = x.shape
        print(d, self.d_model)
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        config_file = self.config_file

        if self.use_flash_attention:
            M = self.device.compute_module.core.SRAM_size

            q = self.Q_proj(x, self.Wq, config_file)
            k = self.K_proj(x, self.Wk, config_file)
            v = self.V_proj(x, self.Wv, config_file)
            q = self.Q_reshape(q, [b, s, h // dev_cnt, d_h])
            k = self.K_reshape(k, [b, s, h // dev_cnt, d_h])
            v = self.V_reshape(v, [b, s, h // dev_cnt, d_h])

            q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
            K_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
            V_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]

            fa_output = self.flash_attention(q_T, K_T, V_T, self.W0, M, s + 1, d, h)
            h0 = self.H_transpose(fa_output, [0, 2, 1, 3])  # [b, 1, h / dev_cnt, d_h]
            h0 = self.H_reshape(h0, [b, s, d // dev_cnt])
            mha_output = self.H_matmul0(h0, self.W0, config_file)  # [b, 1, d]
        else:

            q = self.Q_proj(x, self.Wq, config_file)
            k = self.K_proj(x, self.Wk, config_file)
            v = self.V_proj(x, self.Wv, config_file)

            q = self.Q_reshape(q, [b, s, h // dev_cnt, d_h])
            k = self.K_reshape(k, [b, s, h // dev_cnt, d_h])
            v = self.V_reshape(v, [b, s, h // dev_cnt, d_h])

            q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
            K_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
            V_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]


            a = self.Q_mul_K(q_T, K_T, config_file)  # [b, h / dev_cnt, 1, s+1]
            a_prob = self.A_softmax(a, s + 1, config_file)
            h0 = self.A_mul_V(a_prob, V_T, config_file)  # [b, h / dev_cnt, 1, d_h]
            h0 = self.H_transpose(h0, [0, 2, 1, 3])  # [b, 1, h / dev_cnt, d_h]
            h0 = self.H_reshape(h0, [b, s, d // dev_cnt])
            mha_output = self.H_matmul0(h0, self.W0, config_file)  # [b, 1, d]
        mha_output = self.layer_norm0(mha_output)
        assert mha_output.shape == [b, s, d]
        if dev_cnt > 1:
            mha_output = self.allreduce_mha(mha_output)

#        print("Check:", mha_output.shape)
        # feed-forward network
#        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        h1 = self.H_matmul1(mha_output, self.W1, config_file)  # [b, 1, 4 * d / dev_cnt]
#        print("Check:", h1.shape) # [16,1,8192]
        assert h1.shape == [b, s, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2, config_file)  #  [b, 1, d]
        assert h2.shape == [b, s, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, s, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
        )
        return h2

    def compile_and_simulate(self, system: System):
        device = system.device
        interconnect = system.interconnect
        

        print("collecting layernorm(MHA) tiles")
        mha_layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, 'MHA_collect')
            + device.compute_module.overhead.layernorm
        )

        print("collecting q projection tiles")
        q_proj_latency = (
            self.Q_proj.compile_and_simulate(device, 'q_projection_collect')
            + device.compute_module.overhead.matmul
        )
        
        print("collecting k projection tiles")
        k_proj_latency = (
            self.K_proj.compile_and_simulate(device, 'k_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("collecting v projection tiles")
        v_proj_latency = (
            self.V_proj.compile_and_simulate(device, 'v_projection_collect')
            + device.compute_module.overhead.matmul
        )

        if self.use_flash_attention:
            print("collecting flash attention tiles")
            flash_attention_latency = (
                self.flash_attention.compile_and_simulate(device, 'flash_attention_collect')
                + device.compute_module.overhead.flash_attention
            )

 
        else:
            print("collecting q_mul_k tiles")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(device, 'q_mul_k_collect')
                + device.compute_module.overhead.matmul
            )
            print("collecting softmax tiles")
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, 'softmax_collect')
                + device.compute_module.overhead.softmax
            )
            print("collecting a_mul_v tiles")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(device, 'a_mul_v_collect')
                + device.compute_module.overhead.matmul
            )

        print("collecting w0_projection tiles")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, 'w0_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("collecting layernorm(FFN) tiles")
        ffn_layernorm_latency = (
            self.layer_norm1.compile_and_simulate(device, 'FFN_collect')
            + device.compute_module.overhead.layernorm
        )
        print("collecting w1_projection tiles")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, 'w1_projection_collect')
            + device.compute_module.overhead.matmul
        )
        print("collecting gelu tiles")
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, 'gelu_collect')
            + device.compute_module.overhead.gelu
        )
        print("collecting w2_projection tiles")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, 'w2_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("##################################")
        print("####### Simulation Starts ########")
        print("##################################")


        print()
        print("simulating layernorm(MHA)")

        mha_layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, 'MHA')
            + device.compute_module.overhead.layernorm
        )
        if self.use_flash_attention:
            print()
            print("simulating q projection(flash attention)")
            q_proj_latency = (
                self.Q_proj.compile_and_simulate(device, 'q_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating k projection(flash attention)")
            k_proj_latency = (
                self.K_proj.compile_and_simulate(device, 'k_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating v projection(flash attention)")
            v_proj_latency = (
                self.V_proj.compile_and_simulate(device, 'v_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating flash attention")

            flash_attention_latency = self.flash_attention.compile_and_simulate(device, 'flash_attention')
            + device.compute_module.overhead.flash_attention

            print()
            print("simulating w0_projection")

            h_matmul0_latency = (
                self.H_matmul0.compile_and_simulate(device, 'w0_projection')
                + device.compute_module.overhead.matmul
            )

        else:
            print()
            print("simulating q projection")
            q_proj_latency = (
                self.Q_proj.compile_and_simulate(device, 'q_projection')
                + device.compute_module.overhead.matmul
            )
        
            print()
            print("simulating k projection")
            k_proj_latency = (
                self.K_proj.compile_and_simulate(device, 'k_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating v projection")
            v_proj_latency = (
                self.V_proj.compile_and_simulate(device, 'v_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating q_mul_k")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(device, 'q_mul_k')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating softmax")
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, 'softmax')
                + device.compute_module.overhead.softmax
            )
            print()
            print("simulating a_mul_v")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(device, 'a_mul_v')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating w0_projection")
            h_matmul0_latency = (
                self.H_matmul0.compile_and_simulate(device, 'w0_projection')
                + device.compute_module.overhead.matmul
            )
            print()
        print("simulating all_reduce(MHA)")
        if self.device_count > 1:
            mha_allreduce_latency = self.allreduce_mha.simulate(interconnect)
        else:
            mha_allreduce_latency = 0

        print()
        print("simulating layernorm(FFN)")
        ffn_layernorm_latency = (
            self.layer_norm1.compile_and_simulate(device, 'FFN')
            + device.compute_module.overhead.layernorm
        )
        print()
        print("simulating w1_projection")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, 'w1_projection')
            + device.compute_module.overhead.matmul
        )
        print()
        print("simulating gelu")
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, 'gelu')
            + device.compute_module.overhead.gelu
        )
        print()
        print("simulating w2_projection")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, 'w2_projection')
            + device.compute_module.overhead.matmul
        )

        print()
        print("simulating all_reduce(FFN)")
        if self.device_count > 1:
            ffn_allreduce_latency = self.allreduce_ffn.simulate(interconnect)
        else:
            ffn_allreduce_latency = 0

        if self.use_flash_attention:
            self.latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + mha_layernorm_latency
                + flash_attention_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + ffn_layernorm_latency
                + h1_matmul1_latency
                + gelu_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            print(f"Latency,   {self.latency/device.compute_module.clock_freq*24}")
        else:
            attention_latency = (
                + q_mul_k_latency
                + softmax_latency
                + a_mul_v_latency
            )
            self.latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + mha_layernorm_latency
                + attention_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + ffn_layernorm_latency
                + h1_matmul1_latency
                + gelu_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            linear_latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + q_mul_k_latency
                + a_mul_v_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + h1_matmul1_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            non_linear_latency = (
                mha_layernorm_latency
                + softmax_latency
                + ffn_layernorm_latency
                + gelu_latency
            )
            print(f"Linear cycles, {linear_latency}")
            print(f"NON-linear cycles, {non_linear_latency}")
            print(f"Latency,   {self.latency/device.compute_module.clock_freq*24}")
        return self.latency

class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, system: System, config_file, use_flash_attention=False):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model #hidden dimension
        self.n_heads = n_heads
        self.device_count = device_count
        self.use_flash_attention = use_flash_attention
        self.device = system.device
        self.config_file = config_file

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)

        ## Flash attention operators
#        if self.use_flash_attention:
#            self.flash_attention_blocks = []
#            for i in range(self.n_heads):
#                self.flash_attention_blocks.append(self.init_flash_attention_block(data_type))
        self.MHA_reshape = Reshape(data_type)

        ## multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.flash_attention = FlashAttention(data_type)

        # KV Cache #skkim
        self.K_cache = None
        self.V_cache = None

        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    
    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        # M: sram size skkim

        b, _, d = x.shape
        print(d, self.d_model)
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        config_file = self.config_file

#        config_file = './Tiles/tile_size.json'

        if self.K_cache is None or self.V_cache is None:
            # KV cache
            self.K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
            self.V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        if self.use_flash_attention:
            M = self.device.compute_module.core.SRAM_size
#            mha_output = self.flash_attention(q_T, K_T, V_T, s, M)

#            B_c = math.ceil(M/(4*d_h))
#            B_r = min(math.ceil(M/(4*d_h)), d_h)
#            print("Br, Bc: ", B_r, B_c)
#            print("Check Q block num:", s / B_r)
#            print("Check input Q:", x.shape, self.Wq.shape)
#            print("Check input K:", x.shape, self.Wk.shape)
#            print("Check input V:", x.shape, self.Wv.shape)
#            print("Wq, Wk, Wv: ", self.Wq, self.Wk, self.Wv)
#            print("Data type: ", self.data_type)

            q = self.Q_proj(x, self.Wq, config_file)
            k = self.K_proj(x, self.Wk, config_file)
            v = self.V_proj(x, self.Wv, config_file)
            q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
            k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
            v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])

            q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
            k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
            v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]

            K_T = self.K_concat(self.K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
            assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
            V_T = self.V_concat(self.V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
            assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]

            # Update KV cache
            self.K_cache = K_T
            self.V_cache = V_T


            fa_output = self.flash_attention(q_T, K_T, V_T, self.W0, M, s + 1, d, h)
            print("Check input QKT:", q_T.shape, K_T.shape, V_T.shape, fa_output.shape)
            h0 = self.H_transpose(fa_output, [0, 2, 1, 3])  # [b, 1, h / dev_cnt, d_h]
            h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
            mha_output = self.H_matmul0(h0, self.W0, config_file)  # [b, 1, d]
#            fa_output = self.MHA_reshape(mha_output, [b, 1, d // dev_cnt])

#            h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
#            mha_output = self.H_matmul0(h0, self.W0)  # [b, 1, d]

#            mha_output = self.flash_attention(x, self.Wq, self.Wk, self.Wv, M, s, d_h)
        else:

            q = self.Q_proj(x, self.Wq, config_file)
            k = self.K_proj(x, self.Wk, config_file)
            v = self.V_proj(x, self.Wv, config_file)

            print("Check input Q:", x.shape, self.Wq.shape, q.shape)
            print("Check input K:", x.shape, self.Wk.shape, k.shape)
            print("Check input V:", x.shape, self.Wv.shape, v.shape)
            q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
            k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
            v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])

            q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
            k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
            v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]

            K_T = self.K_concat(self.K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
            assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
            V_T = self.V_concat(self.V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
            assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]

            # Update KV cache
            self.K_cache = K_T
            self.V_cache = V_T

            a = self.Q_mul_K(q_T, K_T, config_file)  # [b, h / dev_cnt, 1, s+1]
            print("Check input QKT:", q_T.shape, K_T.shape, a.shape)
            a_prob = self.A_softmax(a, s + 1, config_file)
            h0 = self.A_mul_V(a_prob, V_T, config_file)  # [b, h / dev_cnt, 1, d_h]
            print("Check input AV:", a_prob.shape, V_T.shape, h0.shape)
            h0 = self.H_transpose(h0, [0, 2, 1, 3])  # [b, 1, h / dev_cnt, d_h]
            h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
            mha_output = self.H_matmul0(h0, self.W0, config_file)  # [b, 1, d]
            print("Check output:", mha_output.shape) #[16,1,2048]
        mha_output = self.layer_norm0(mha_output)
#        print("Check:", mha_output.shape)  # [16,1,2048]
        assert mha_output.shape == [b, 1, d]
        if dev_cnt > 1:
            mha_output = self.allreduce_mha(mha_output)

#        print("Check:", mha_output.shape)
        # feed-forward network
#        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        h1 = self.H_matmul1(mha_output, self.W1, config_file)  # [b, 1, 4 * d / dev_cnt]
#        print("Check:", h1.shape) # [16,1,8192]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2, config_file)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + self.K_cache.size * self.K_cache.data_type.word_size
            + self.V_cache.size * self.V_cache.data_type.word_size
        )
        return h2

    def compile_and_simulate(self, system: System):
        device = system.device
        interconnect = system.interconnect
        

        print("collecting layernorm(MHA) tiles")
        mha_layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, 'MHA_collect')
            + device.compute_module.overhead.layernorm
        )

        print("collecting q projection tiles")
        q_proj_latency = (
            self.Q_proj.compile_and_simulate(device, 'q_projection_collect')
            + device.compute_module.overhead.matmul
        )
        
        print("collecting k projection tiles")
        k_proj_latency = (
            self.K_proj.compile_and_simulate(device, 'k_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("collecting v projection tiles")
        v_proj_latency = (
            self.V_proj.compile_and_simulate(device, 'v_projection_collect')
            + device.compute_module.overhead.matmul
        )

        if self.use_flash_attention:
            print("collecting flash attention tiles")
            flash_attention_latency = (
                self.flash_attention.compile_and_simulate(device, 'flash_attention_collect')
                + device.compute_module.overhead.flash_attention
            )

 
        else:
            print("collecting q_mul_k tiles")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(device, 'q_mul_k_collect')
                + device.compute_module.overhead.matmul
            )
            print("collecting softmax tiles")
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, 'softmax_collect')
                + device.compute_module.overhead.softmax
            )
            print("collecting a_mul_v tiles")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(device, 'a_mul_v_collect')
                + device.compute_module.overhead.matmul
            )

        print("collecting w0_projection tiles")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, 'w0_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("collecting layernorm(FFN) tiles")
        ffn_layernorm_latency = (
            self.layer_norm1.compile_and_simulate(device, 'FFN_collect')
            + device.compute_module.overhead.layernorm
        )
        print("collecting w1_projection tiles")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, 'w1_projection_collect')
            + device.compute_module.overhead.matmul
        )
        print("collecting gelu tiles")
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, 'gelu_collect')
            + device.compute_module.overhead.gelu
        )
        print("collecting w2_projection tiles")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, 'w2_projection_collect')
            + device.compute_module.overhead.matmul
        )

        print("##################################")
        print("####### Simulation Starts ########")
        print("##################################")


        print()
        print("simulating layernorm(MHA)")

        mha_layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, 'MHA')
            + device.compute_module.overhead.layernorm
        )
        if self.use_flash_attention:
            print()
            print("simulating q projection(flash attention)")
            q_proj_latency = (
                self.Q_proj.compile_and_simulate(device, 'q_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating k projection(flash attention)")
            k_proj_latency = (
                self.K_proj.compile_and_simulate(device, 'k_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating v projection(flash attention)")
            v_proj_latency = (
                self.V_proj.compile_and_simulate(device, 'v_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating flash attention")

            flash_attention_latency = self.flash_attention.compile_and_simulate(device, 'flash_attention')
            + device.compute_module.overhead.flash_attention

            print()
            print("simulating w0_projection")

            h_matmul0_latency = (
                self.H_matmul0.compile_and_simulate(device, 'w0_projection')
                + device.compute_module.overhead.matmul
            )

        else:
            print()
            print("simulating q projection")
            q_proj_latency = (
                self.Q_proj.compile_and_simulate(device, 'q_projection')
                + device.compute_module.overhead.matmul
            )
        
            print()
            print("simulating k projection")
            k_proj_latency = (
                self.K_proj.compile_and_simulate(device, 'k_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating v projection")
            v_proj_latency = (
                self.V_proj.compile_and_simulate(device, 'v_projection')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating q_mul_k")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(device, 'q_mul_k')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating softmax")
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, 'softmax')
                + device.compute_module.overhead.softmax
            )
            print()
            print("simulating a_mul_v")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(device, 'a_mul_v')
                + device.compute_module.overhead.matmul
            )
            print()
            print("simulating w0_projection")
            h_matmul0_latency = (
                self.H_matmul0.compile_and_simulate(device, 'w0_projection')
                + device.compute_module.overhead.matmul
            )
            print()
        print("simulating all_reduce(MHA)")
        if self.device_count > 1:
            mha_allreduce_latency = self.allreduce_mha.simulate(interconnect)
        else:
            mha_allreduce_latency = 0

        print()
        print("simulating layernorm(FFN)")
        ffn_layernorm_latency = (
            self.layer_norm1.compile_and_simulate(device, 'FFN')
            + device.compute_module.overhead.layernorm
        )
        print()
        print("simulating w1_projection")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, 'w1_projection')
            + device.compute_module.overhead.matmul
        )
        print()
        print("simulating gelu")
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, 'gelu')
            + device.compute_module.overhead.gelu
        )
        print()
        print("simulating w2_projection")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, 'w2_projection')
            + device.compute_module.overhead.matmul
        )

        print()
        print("simulating all_reduce(FFN)")
        if self.device_count > 1:
            ffn_allreduce_latency = self.allreduce_ffn.simulate(interconnect)
        else:
            ffn_allreduce_latency = 0

        if self.use_flash_attention:
            self.latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + mha_layernorm_latency
                + flash_attention_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + ffn_layernorm_latency
                + h1_matmul1_latency
                + gelu_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            print(f"Latency,   {self.latency/device.compute_module.clock_freq*24}")
        else:
            attention_latency = (
                + q_mul_k_latency
                + softmax_latency
                + a_mul_v_latency
            )
            self.latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + mha_layernorm_latency
                + attention_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + ffn_layernorm_latency
                + h1_matmul1_latency
                + gelu_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            linear_latency = (
                q_proj_latency
                + k_proj_latency
                + v_proj_latency
                + q_mul_k_latency
                + a_mul_v_latency
                + h_matmul0_latency
                + mha_allreduce_latency
                + h1_matmul1_latency
                + h2_matmul2_latency
                + ffn_allreduce_latency
            )
            non_linear_latency = (
                mha_layernorm_latency
                + softmax_latency
                + ffn_layernorm_latency
                + gelu_latency
            )
            print(f"Linear cycles, {linear_latency}")
            print(f"NON-linear cycles, {non_linear_latency}")
            print(f"Latency,   {self.latency/device.compute_module.clock_freq*24}")
        return self.latency

