from utils import size
from typing import List, Tuple
from hardware_model.hw_device import Device
from software_model.sw_operators import Operator
from software_model.sw_utils import Tensor, DataType
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
import math
from software_model.sw_softmax import Softmax

class FlashAttention(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
#        self.x_shape = None
#        self.Wq_shape = None
#        self.Wk_shape = None
#        self.Wv_shape = None
        self.Query = None
        self.Key = None
        self.Value = None
        self.W0 = None
        self.flop_count = 0
        self.io_count = 0
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

#    def __call__(self, x: Tensor, Wq: Tensor, Wk: Tensor, Wv: Tensor, sram_size:int, seq_len: int, head_dim: int) -> Tensor:
    def __call__(self, input_q: Tensor, input_k: Tensor, input_v: Tensor, input_w0: Tensor, sram_size:int, seq_len: int, dim: int, head: int) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        self.seq_len = seq_len # prompt + KV cache + input(1) = 1024 + 2048 + 1 -> 3073
        self.d = dim
        self.d_h = dim // head
        # Set block sizes
#        self.B_c = math.ceil(sram_size / (4 * self.d))  # Ensure block size fits in SRAM -> 1024
#        self.B_r = min(math.ceil(sram_size / (4 * self.d)), self.d)  # min(1024,128) -> 128
        self.B_c = math.ceil(sram_size / (4 * self.d)) # -> 64
        self.B_r = self.d_h

        # Divide matrices into blocks
        self.T_c = math.ceil(input_k.shape[-1] / self.B_c)  # Number of column blocks
        self.T_r = math.ceil(input_q.shape[-2] / self.B_r)  # Number of row blocks


        self.Query = input_q
        self.Key = input_k
        self.Value = input_v
        self.W0 = input_w0
        self.output_shape = input_q.shape
        flops = 0
        
#        print("Check in block size:", self.B_c, self.B_r, self.T_c, self.T_r, self.seq_len)
#        print("Check in flash attention - QKV:", self.Query.shape, self.Key.shape, self.Value.shape)
        for j in range(self.T_c):
            for i in range(self.T_r):

                # Load blocks Q_i, O_i, l_i, and m_i into SRAM
                self.M = self.Query.shape[-2]
                self.N = self.Key.shape[-1]
                self.K = self.Key.shape[-2]

#                print(f'#SKKIM flash sttention dimension: {self.M}, {self.N}, {self.K}')

                # TODO: skkim - 1. call computational_graph, 2. flop_count, 3. io_count for softmax, SV, W0_projection 
#                self.computational_graph = self.ComputationalGraph(
#                    self.M, self.N, self.K, self.data_type
#                )
                self.flop_count += 2 * self.M * self.K * self.N
                self.io_count += self.M * self.K + self.K * self.N + self.M * self.N

        output = Tensor(self.output_shape, self.data_type)
        return output

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k
        

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-"*10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    class softmax_Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            is_l1_double_buffering: bool = False,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.is_l1_double_buffering = is_l1_double_buffering

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, is_l2_double_buffering: {self.is_l2_double_buffering}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, is_l1_double_buffering: {self.is_l1_double_buffering}"
            )

    class matmul_Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
            )
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )
            print(
                f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
            )


    def split_to_tile(self,n, unit):
        result = []
        while n >= unit:
            result.append(unit)
            n -= unit
        if n > 0:
            result.append(n)
        return result

    def compile_and_simulate(
        self,
        pcb_module: Device,
        ops_name : str,
    ):

        qkT_latency = 0
        softmax_latency = 0
        sv_latency = 0
        flash_attention_tot_cycle = 0
        flash_attention_io_cycle = 0
        flash_attention_compute_cycle = 0
        flash_attention_time_tick = 0

        previous_compute_cycle = 0
#        w0_latency = 0
        Q_i_list = self.split_to_tile(self.Query.shape[-2], self.B_r)
        K_j_list = self.split_to_tile(self.Key.shape[-1],self.B_c)
        V_j_list = self.split_to_tile(self.Value.shape[-2],self.B_c)

        for j in range(self.T_c):
            # Load blocks K_j and V_j into SRAM
            K_j = self.Key.copy()
            K_j.shape[-1] = K_j_list[j]
            V_j = self.Value.copy()
            V_j.shape[-2] = V_j_list[j]
#            print("K_j:", K_j)
            for i in range(self.T_r):
                Q_i = self.Query.copy()
                Q_i.shape[-2] = Q_i_list[i]

                self.M = Q_i.shape[-2]
                self.N = K_j.shape[-1]
                self.K = K_j.shape[-3] * K_j.shape[-2]
#                print("SKKIM QKT size(M,N,K)", self.M, self.N, self.K)
#                print("SKKIM copy test", self.Key, self.Key.copy(), K_j)
#                print("SKKIM QKV (Q,K,V)", self.Query.shape, self.Key.shape, self.Value.shape)
#                print("SKKIM QKV tile (Q,K,V)", Q_i.shape, K_j.shape, V_i.shape)
                self.computational_graph = self.ComputationalGraph(
                    self.M, self.N, self.K, self.data_type
                )
                tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_qkT_latency, tmp_sa_util_rate = self.matmul_compile_and_simulate(pcb_module, 'projection_and_qk')
#                print(tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_qkT_latency)
# skkim: do not consider first io cycle(1 read, 0 compute)
###################################################QKT############################################################################
                flash_attention_tot_cycle += ( max(tmp_read_cycle - previous_compute_cycle , 0) + tmp_compute_cycle)
                flash_attention_io_cycle += tmp_read_cycle 
                flash_attention_compute_cycle += tmp_compute_cycle


                skkim_double_buffering = True

                if skkim_double_buffering: flash_attention_time_tick += tmp_compute_cycle
                else: flash_attention_time_tick += (tmp_read_cycle + tmp_compute_cycle)
                previous_compute_cycle = flash_attention_compute_cycle

                remained_io_cycle = max(tmp_read_cycle - tmp_compute_cycle, 0)

                print("total cycle(X) :", flash_attention_time_tick)
                print("compute cycle(X1) :", tmp_compute_cycle)
                print("io cycle(X2) :", tmp_read_cycle)
                if skkim_double_buffering: print("current cycle(X3) :", tmp_compute_cycle)
                else: print("current cycle(X3) :", (tmp_read_cycle + tmp_compute_cycle))
                print("sram occupancy[%](Y2) : ",  (tmp_read_byte)/pcb_module.compute_module.core.SRAM_size * 100)
                print("memory bw util[%](Y1) : ", 100 )
                print("sa util[%](Y3) : ", tmp_sa_util_rate * 100)
                print("va util[%](Y3) : ", 0 )

 #               print("check softmax")
##################################################softmax###########################################################################
#                print("SKKIM softmax size(M,N)", self.M, self.N)
                _, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_softmax_latency, tmp_va_util_rate = self.softmax_compile_and_simulate(pcb_module, 'mha_softmax')
#                print(tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_softmax_latency)
                flash_attention_tot_cycle += tmp_compute_cycle
                flash_attention_compute_cycle += tmp_compute_cycle
                flash_attention_time_tick += tmp_compute_cycle
                remained_io_cycle = max(remained_io_cycle - tmp_compute_cycle, 0)

                print("total cycle(X) :", flash_attention_time_tick)
                print("compute cycle(X1) :", tmp_compute_cycle)
                print("io cycle(X2) :", 0)
                print("current cycle(X3) :", tmp_compute_cycle)
                print("sram occupancy[%](Y2) : ",  (tmp_read_byte)/pcb_module.compute_module.core.SRAM_size * 100)
                print("memory bw util[%](Y1) : ", 0 )
                print("sa util[%](Y3) : ", 0)
                print("va util[%](Y3) : ", tmp_va_util_rate * 100 )

##########################################################SV########################################################################
#                print("check sv")
                #self.M = Q_i.shape[2]
                #self.N = V_j.shape[-1]
                #self.K = K_j.shape[-1] * Q_i.shape[0] * Q_i.shape[1]
                self.M = Q_i.shape[-2]
                self.N = V_j.shape[-1]
                self.K = V_j.shape[-3] * V_j.shape[-2]
                self.computational_graph = self.ComputationalGraph(
                    self.M, self.N, self.K, self.data_type
                )
#                print("SKKIM SV size(M,N,K)", self.M, self.N, self.K)
                _, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_sv_latency, tmp_sa_util_rate = self.matmul_compile_and_simulate(pcb_module, 'sv')
#                print(tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_sv_latency)
                flash_attention_tot_cycle += (tmp_compute_cycle + tmp_write_cycle)
                flash_attention_io_cycle += tmp_write_cycle 
                flash_attention_compute_cycle += tmp_compute_cycle
                remained_io_cycle = max(remained_io_cycle - tmp_compute_cycle, 0)

                if skkim_double_buffering:  flash_attention_time_tick += (remained_io_cycle + tmp_compute_cycle + tmp_write_cycle)
                else: flash_attention_time_tick += (tmp_compute_cycle + tmp_write_cycle)
                previous_compute_cycle = flash_attention_compute_cycle

                print("total cycle(X) :", flash_attention_time_tick)
                print("compute cycle(X1) :", tmp_compute_cycle)
                print("io cycle(X2) :", tmp_write_cycle)
                if skkim_double_buffering:  print("current cycle(X3) :", (remained_io_cycle + tmp_compute_cycle +  tmp_write_cycle))
                else: print("current cycle(X3) :", (tmp_compute_cycle +  tmp_write_cycle))
                print("sram occupancy[%](Y2) : ",  (tmp_read_byte)/pcb_module.compute_module.core.SRAM_size * 100)
                print("memory bw util[%](Y1) : ", 100 )
                print("sa util[%](Y3) : ", tmp_sa_util_rate * 100)
                print("va util[%](Y3) : ", 0 )

#                print("memory bw util[%](Y1) :", (skkim_io_byte/skkim_io_cycle)/chiplet_module.compute_module.l2_bandwidth_per_cycle * 100)
#                print("sram occupancy[%](Y2) :", ( current_batch_M_K_read_count+ current_batch_K_N_read_count +  current_batch_M_N_read_count) * data_type.word_size * 2 / chiplet_module.compute_module.core.SRAM_size * 100) # double buffering = True
#                print("sa util[%](Y3) : ", util_rate)

                flash_attention_tot_cycle = 0
                flash_attention_io_cycle = 0
                flash_attention_compute_cycle = 0
                '''
                print("check w0_proj")
                self.M = size(Q_i.shape[:-1])
                self.K = V_j.shape[-1]
                self.N = self.W0.shape[-1]
                self.computational_graph = self.ComputationalGraph(
                    self.M, self.N, self.K, self.data_type
                )
 
                tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_w0_latency = self.matmul_compile_and_simulate(pcb_module, 'w0_proj')
                print(tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_w0_latency)
                '''

                qkT_latency += tmp_qkT_latency
                softmax_latency = tmp_softmax_latency
                sv_latency += tmp_sv_latency
#                w0_latency += tmp_w0_latency

        flash_attention_time_tick += (tmp_compute_cycle + tmp_write_cycle)
        print("total cycle(X) :", flash_attention_time_tick)
        print("compute cycle(X1) :", tmp_compute_cycle)
        print("io cycle(X2) :", tmp_write_cycle)
        print("current cycle(X3) :", (tmp_compute_cycle +  tmp_write_cycle))
        print("sram occupancy[%](Y2) : ",  (tmp_read_byte)/pcb_module.compute_module.core.SRAM_size * 100)
        print("memory bw util[%](Y1) : ", 100 )
        print("sa util[%](Y3) : ", tmp_sa_util_rate * 100)
        print("va util[%](Y3) : ", 0 )


        return qkT_latency + softmax_latency + sv_latency

    def matmul_compile_and_simulate(
        self,
        pcb_module: Device,
        ops_name : str,
    ):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K

        l2_tile_M = self.computational_graph.M
        l2_tile_N = self.computational_graph.N
        l2_tile_K = self.computational_graph.K

        is_l2_double_buffering = False

        l1_tile_M = self.computational_graph.M
        l1_tile_N = self.computational_graph.N
        l1_tile_K = self.computational_graph.K
        next_ops_name = 'w1_projection'

        l2_loop_order = "knm"
        l1_loop_order = "knm"
        for (
            l0_M_tiling_factor,
            l0_N_tiling_factor,
            l0_K_tiling_factor,
        ) in [(1, 1, 1)]:
#                    ) in [(1, 2, 1)]: #skkim
            mapping = self.matmul_Mapping(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                is_l2_double_buffering,
                l1_tile_M,
                l1_tile_N,
                l1_tile_K,
                l2_loop_order,
                l1_loop_order,
                l0_M_tiling_factor,
                l0_N_tiling_factor,
                l0_K_tiling_factor,
            )
#                        mapping.display()
            # start=time.time()
#            cycle_count = self.matmul_simulate(
            read_byte, read_cycle, compute_cycle, write_byte, write_cycle, cycle_count, sa_util_rate = self.matmul_simulate(
                self.computational_graph,
                mapping,
                pcb_module,
                ops_name,
                next_ops_name
            )

            # end=time.time()
            # print(f'simulation time: {end-start}')
#            print("compare cycle count, min cycle count: ", cycle_count, min_cycle_count)
            if cycle_count < min_cycle_count:
                min_cycle_count = cycle_count
                best_mapping = mapping
                best_read_byte = read_byte
                best_read_cycle = read_cycle
                best_compute_cycle = compute_cycle
                best_write_byte = write_byte
                best_write_cycle = write_cycle
                best_sa_util_rate = sa_util_rate
#            print("compare done. cycle count, mapping: ", min_cycle_count, best_mapping)
        self.best_mapping = best_mapping
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
        K_size = self.best_mapping.l1_tile_K
#        print("M,N,K: ", M_size, N_size, K_size)
        if mapping.is_l2_double_buffering:
            occupancy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size * 2 / pcb_module.compute_module.core.SRAM_size * 100
        else:
            occupancy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size / pcb_module.compute_module.core.SRAM_size * 100
#        print("CCCHECK",occupancy,M_size, N_size,K_size,self.data_type.word_size,pcb_module.compute_module.core.SRAM_size)
#        print("Tile size, ",M_size, N_size, K_size)
#        print("Word size, ",self.data_type.word_size)
#        print("SRAM size, ",pcb_module.compute_module.core.SRAM_size)
#        print("IO BW, ",pcb_module.compute_module.l2_bandwidth_per_cycle)
#        print("SRAM Util, ",occupancy)
#        print("Min Cycle, ",min_cycle_count)
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.read_byte = best_read_byte
        self.read_cycle = best_read_cycle
        self.compute_cycle = best_compute_cycle
        self.write_byte = best_write_byte
        self.write_cycle = best_write_cycle
        self.sa_util_rate = best_sa_util_rate

        #self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.best_latency = min_cycle_count
        self.latency = self.best_latency
        self.best_mapping.display()

#        print("Occupancy(%)/sram:",occupancy,"/",pcb_module.compute_module.core.SRAM_size)
        return self.read_byte, self.read_cycle, self.compute_cycle, self.write_byte, self.write_cycle, self.latency, self.sa_util_rate

    def matmul_simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: matmul_Mapping,
        pcb_module: Device,
        ops_name: str,
        next_ops_name: str,
    ) -> tuple:
        # Initialize lookup table if not already loaded
        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"], inplace=True
            )

        # Extract computational graph parameters
        M, N, K = computational_graph.M, computational_graph.N, computational_graph.K
        data_type = computational_graph.data_type

        # Extract L2 tile dimensions
        l2_tile_M, l2_tile_N, l2_tile_K = mapping.l2_tile_M, mapping.l2_tile_N, mapping.l2_tile_K

        # Verify L2 memory constraints
        l2_memory_usage = (
            l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
        )
        l2_memory_limit = (
            pcb_module.compute_module.l2_size // self.data_type.word_size
            // (2 if mapping.is_l2_double_buffering else 1)
        )
        assert l2_memory_usage <= l2_memory_limit, "L2 memory constraint violated"

        # Determine tile counts and remainders
        M_l2_t, N_l2_t, K_l2_t = M // l2_tile_M, N // l2_tile_N, K // l2_tile_K
        M_remain, N_remain, K_remain = M % l2_tile_M, N % l2_tile_N, K % l2_tile_K

        # Initialize L2 tiles
        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.matmul_L2TileSimulator,
        )

        # Populate L2 tiles
        def initialize_tile(m, n, k, tile_M, tile_N, tile_K):
            l2_tiles[m, n, k] = self.matmul_L2TileSimulator(
                tile_M,
                tile_N,
                tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )

        for m in range(M_l2_t):
            for n in range(N_l2_t):
                for k in range(K_l2_t):
                    initialize_tile(m, n, k, l2_tile_M, l2_tile_N, l2_tile_K)

        if M_remain:
            for n in range(N_l2_t):
                for k in range(K_l2_t):
                    initialize_tile(M_l2_t, n, k, M_remain, l2_tile_N, l2_tile_K)
        if N_remain:
            for m in range(M_l2_t):
                for k in range(K_l2_t):
                    initialize_tile(m, N_l2_t, k, l2_tile_M, N_remain, l2_tile_K)
        if K_remain:
            for m in range(M_l2_t):
                for n in range(N_l2_t):
                    initialize_tile(m, n, K_l2_t, l2_tile_M, l2_tile_N, K_remain)
        if M_remain and N_remain:
            for k in range(K_l2_t):
                initialize_tile(M_l2_t, N_l2_t, k, M_remain, N_remain, l2_tile_K)
        if M_remain and K_remain:
            for n in range(N_l2_t):
                initialize_tile(M_l2_t, n, K_l2_t, M_remain, l2_tile_N, K_remain)
        if N_remain and K_remain:
            for m in range(M_l2_t):
                initialize_tile(m, N_l2_t, K_l2_t, l2_tile_M, N_remain, K_remain)
        if M_remain and N_remain and K_remain:
            initialize_tile(M_l2_t, N_l2_t, K_l2_t, M_remain, N_remain, K_remain)

        # Initialize performance metrics
        total_cycle_count = 0
        skkim_read_byte = 0
        skkim_read_cycle_count = 0
        skkim_compute_cycle_count = 0
        skkim_write_byte = 0
        skkim_write_cycle_count = 0
        skkim_sa_util_rate = 0

        # Process tiles
        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K), mapping.l2_loop_order
        ):
            current_tile = l2_tiles[m, n, k]
#            print(f"Processing Tile ({m}, {n}, {k}):")
#            print(f"  Read Byte: {current_tile.total_read_byte}")
#            print(f"  Write Byte: {current_tile.total_write_byte}")
#            print(f"  Compute Cycle Count: {current_tile.total_compute_cycle_count}")

            current_tile_read_cycles = (
                current_tile.M_K_io_cycle_count + current_tile.K_N_io_cycle_count
            )

            if k > 0:
                current_tile_read_cycles += current_tile.M_N_io_cycle_count

            current_tile_compute_cycles = current_tile.compute_cycle_count
            if k > 0:
                current_tile_compute_cycles += current_tile.K_reduction_cycle_count

            current_tile_write_cycles = (
                0 if (m == 0 and n == 0) else current_tile.M_N_io_cycle_count
            )

            if mapping.is_l2_double_buffering:
                total_cycle_count += max(current_tile_read_cycles, current_tile_compute_cycles)
            else:
                total_cycle_count += (
                    current_tile_read_cycles + current_tile_compute_cycles + current_tile_write_cycles
                )

            skkim_read_byte += current_tile.total_read_byte
            skkim_read_cycle_count += current_tile.total_read_cycle_count
            skkim_compute_cycle_count += current_tile.total_compute_cycle_count
            skkim_write_byte += current_tile.total_write_byte
            skkim_write_cycle_count += current_tile.total_write_cycle_count
            skkim_sa_util_rate = current_tile.skkim_sa_util_rate

        # Add final tile cycles
        final_tile = l2_tiles[-1, -1, -1]
#        print("Final Tile:")
#        print(f"  Read Byte: {final_tile.total_read_byte}")
#        print(f"  Write Byte: {final_tile.total_write_byte}")
#        print(f"  Compute Cycle Count: {final_tile.total_compute_cycle_count}")

        total_cycle_count += (
            final_tile.M_N_io_cycle_count + final_tile.compute_cycle_count
        )
        if K > l2_tile_K:
            total_cycle_count += final_tile.K_reduction_cycle_count

        # Return results
        return (
            skkim_read_byte,
            skkim_read_cycle_count,
            skkim_compute_cycle_count,
            skkim_write_byte,
            skkim_write_cycle_count,
            total_cycle_count,
            skkim_sa_util_rate,
        )


    def softmax_compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.K // self.d_h
        N = self.computational_graph.N

        data_type = self.computational_graph.data_type
        is_l2_double_buffering = False
        is_l1_double_buffering = False

        l2_tile_N = N
        l2_tile_M = M

        l1_tile_N = N
        l1_tile_M  = M 

        if is_l1_double_buffering:
            assert (
                l1_tile_M * l1_tile_N * data_type.word_size
                <= pcb_module.compute_module.core.SRAM_size // 2
            )
        else:
            assert (
                l1_tile_M * l1_tile_N * data_type.word_size
                <= pcb_module.compute_module.core.SRAM_size
            )
        mapping = self.softmax_Mapping(
            l2_tile_M,
            l2_tile_N,
            is_l2_double_buffering,
            l1_tile_M,
            l1_tile_N,
            is_l1_double_buffering,
        )
        read_byte, read_cycle, compute_cycle, write_byte, write_cycle, cycle_count, va_util_rate = self.softmax_simulate(
            self.computational_graph, mapping, pcb_module
        )

        if cycle_count < min_cycle_count:
            min_cycle_count = cycle_count
            best_mapping = mapping
            best_read_byte = read_byte
            best_read_cycle = read_cycle
            best_compute_cycle = compute_cycle
            best_write_byte = write_byte
            best_write_cycle = write_cycle
            best_va_util_rate = va_util_rate

        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.read_byte = best_read_byte
        self.read_cycle = best_read_cycle
        self.compute_cycle = best_compute_cycle
        self.write_byte = best_write_byte
        self.write_cycle = best_write_cycle

        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        self.va_util_rate = best_va_util_rate
        # self.best_mapping.display()
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
#        print("Tile size, ",M_size, N_size)
        
        return self.read_byte, self.read_cycle, self.compute_cycle, self.write_byte, self.write_cycle, self.latency, self.va_util_rate

    def softmax_simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: softmax_Mapping,
        pcb_module: Device,
    ) -> int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * N * data_type.word_size * 2
                <= pcb_module.compute_module.l2_size
            )
        else:
            assert (
                l2_tile_M * N * data_type.word_size <= pcb_module.compute_module.l2_size
            )

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.softmax_L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.softmax_L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.softmax_L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

        # Initialize performance metrics
        current_tile = l2_tiles[0]

        total_cycle_count = current_tile.compute_cycle_count

        skkim_read_byte = current_tile.total_read_byte
        skkim_read_cycle_count = current_tile.total_read_cycle_count
        skkim_compute_cycle_count = current_tile.total_compute_cycle_count
        skkim_write_byte = current_tile.total_write_byte
        skkim_write_cycle_count = current_tile.total_write_cycle_count
        skkim_va_util_rate = current_tile.skkim_va_util_rate

        return (
            skkim_read_byte,
            skkim_read_cycle_count,
            skkim_compute_cycle_count,
            skkim_write_byte,
            skkim_write_cycle_count,
            total_cycle_count,
            skkim_va_util_rate,
        )




    class softmax_L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            total_read_byte, total_read_cycle_count, total_compute_cycle_count, total_write_byte, total_write_cycle_count, skkim_total_cycle_count, skkim_va_util_rate = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_cycle_count = skkim_total_cycle_count
            self.total_read_byte = total_read_byte
            self.total_read_cycle_count = total_read_cycle_count
            self.total_compute_cycle_count = total_compute_cycle_count
            self.total_write_byte = total_write_byte
            self.total_write_cycle_count = total_write_cycle_count
            self.skkim_total_cycle_count = skkim_total_cycle_count
            self.skkim_va_util_rate = skkim_va_util_rate


#self.read_cycle_count, self.write_cycle_count, self.compute_cycle_count = 0

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = FlashAttention.softmax_L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)           
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count
                + l1_tile.write_cycle_count
                + l1_tile.compute_cycle_count
            )
            l1_tile_compute_cycle_count = (
                l1_tile.compute_cycle_count
            )
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
                l1_tile_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )

            return (
                l1_tile.read_byte,
                l1_tile.read_cycle_count,
                l1_tile.compute_cycle_count,
                l1_tile.write_byte,
                l1_tile.write_cycle_count,
                l1_tile.read_cycle_count + l1_tile.write_cycle_count + l1_tile.compute_cycle_count,
                l1_tile.va_util,
            )



    class matmul_L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
            ops_name : str,
            next_ops_name : str,
        ):
            # print(f'L2 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.K_reduction_cycle_count = ceil(
                M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )
            self.K_reduction_io_count = 2 * M * N * data_type.word_size
            self.M_K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, K, data_type, pcb_module
            )
            self.K_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                K, N, data_type, pcb_module
            )
            self.M_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
#            print("#####################", self.M_K_io_cycle_count, self.K_N_io_cycle_count, self.M_N_io_cycle_count)
#            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
            total_read_byte, total_read_cycle_count, total_compute_cycle_count, total_write_byte, total_write_cycle_count, skkim_total_cycle_count, skkim_sa_util_rate = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name
            )

            self.compute_cycle_count = skkim_total_cycle_count
            self.total_read_byte = total_read_byte
            self.total_read_cycle_count = total_read_cycle_count
            self.total_compute_cycle_count = total_compute_cycle_count
            self.total_write_byte = total_write_byte
            self.total_write_cycle_count = total_write_cycle_count
            self.skkim_total_cycle_count = skkim_total_cycle_count
            self.skkim_sa_util_rate = skkim_sa_util_rate

            #print("class Matmul > def compile_and_simulate > def simulate > class L2TileSimulator > def __init__")
            #print("result of simulate_l2_tile_io_cycle_count: ")
            #print(self.M_K_io_cycle_count)
            #print(self.K_N_io_cycle_count)
            #print(self.M_N_io_cycle_count)
            #print("result of simulate_l2_tile_compute_cycle_count: ")
            #print(self.compute_cycle_count)

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type,
            mapping,
            chiplet_module,
            look_up_table,
            ops_name: str,
        ) -> tuple:
            # Extract L1 tile dimensions
            l1_tile_M, l1_tile_N, l1_tile_K = mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K
            
            # Calculate L1 tile configurations
            M_l1_t, N_l1_t, K_l1_t = M // l1_tile_M, N // l1_tile_N, K // l1_tile_K
            M_remain, N_remain, K_remain = M % l1_tile_M, N % l1_tile_N, K % l1_tile_K

            # Initialize L1 tiles
            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=object,
            )

            def initialize_tile(m, n, k, tile_M, tile_N, tile_K):
                l1_tiles[m, n, k] = FlashAttention.matmul_L1TileSimulator(
                    tile_M, tile_N, tile_K, data_type, mapping, chiplet_module, look_up_table
                )

            # Populate L1 tiles
            for m in range(M_l1_t):
                for n in range(N_l1_t):
                    for k in range(K_l1_t):
                        initialize_tile(m, n, k, l1_tile_M, l1_tile_N, l1_tile_K)

            if M_remain:
                for n in range(N_l1_t):
                    for k in range(K_l1_t):
                        initialize_tile(M_l1_t, n, k, M_remain, l1_tile_N, l1_tile_K)
            if N_remain:
                for m in range(M_l1_t):
                    for k in range(K_l1_t):
                        initialize_tile(m, N_l1_t, k, l1_tile_M, N_remain, l1_tile_K)
            if K_remain:
                for m in range(M_l1_t):
                    for n in range(N_l1_t):
                        initialize_tile(m, n, K_l1_t, l1_tile_M, l1_tile_N, K_remain)

            if M_remain and N_remain:
                for k in range(K_l1_t):
                    initialize_tile(M_l1_t, N_l1_t, k, M_remain, N_remain, l1_tile_K)
            if M_remain and K_remain:
                for n in range(N_l1_t):
                    initialize_tile(M_l1_t, n, K_l1_t, M_remain, l1_tile_N, K_remain)
            if N_remain and K_remain:
                for m in range(M_l1_t):
                    initialize_tile(m, N_l1_t, K_l1_t, l1_tile_M, N_remain, K_remain)
            if M_remain and N_remain and K_remain:
                initialize_tile(M_l1_t, N_l1_t, K_l1_t, M_remain, N_remain, K_remain)

            # Tile size calculations
            def calculate_tile_sizes(dim_1, dim_2, tile_size_1, tile_size_2, remain_1, remain_2):
                tile_sizes = np.zeros([ceil(dim_1 / tile_size_1), ceil(dim_2 / tile_size_2)], dtype=int)
                tile_sizes[:dim_1 // tile_size_1, :dim_2 // tile_size_2] = tile_size_1 * tile_size_2
                if remain_1:
                    tile_sizes[-1, :dim_2 // tile_size_2] = remain_1 * tile_size_2
                if remain_2:
                    tile_sizes[:dim_1 // tile_size_1, -1] = tile_size_1 * remain_2
                if remain_1 and remain_2:
                    tile_sizes[-1, -1] = remain_1 * remain_2
                return tile_sizes

            M_K_tile_size = calculate_tile_sizes(M, K, l1_tile_M, l1_tile_K, M_remain, K_remain)
            K_N_tile_size = calculate_tile_sizes(K, N, l1_tile_K, l1_tile_N, K_remain, N_remain)
            M_N_tile_size = calculate_tile_sizes(M, N, l1_tile_M, l1_tile_N, M_remain, N_remain)

            # Cycle count variables
            total_read_byte = 0
            total_read_cycle_count = 0
            total_compute_cycle_count = 0
            total_write_byte = 0
            total_write_cycle_count = 0
            skkim_total_cycle_count = 0


            # Active L1 tile simulation
            for m, n, k in FlashAttention.generate_tile_loops(
                ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K), mapping.l1_loop_order
            ):
                active_tiles = [(m, n, k, l1_tiles[m, n, k])]

                current_batch_M_K_read_count = 0
                current_batch_K_N_read_count = 0
                current_batch_M_N_read_count = 0
                current_batch_compute_cycle_count = 0

                for tile in active_tiles:
                    tile_m, tile_n, tile_k, tile_obj = tile
                    current_batch_M_K_read_count += tile_obj.M * tile_obj.K
                    current_batch_K_N_read_count += tile_obj.K * tile_obj.N
                    current_batch_M_N_read_count += tile_obj.M * tile_obj.N

                    # Compute cycle count calculation
                    temp_l1_tile_compute_cycle_count = tile_obj.compute_cycle_count
                    if tile_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            tile_obj.M
                            * tile_obj.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count, temp_l1_tile_compute_cycle_count
                    )

                # Calculate total read count
                current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
                )

                # Update total bytes
                total_read_byte += current_batch_read_count * data_type.word_size
                total_write_byte += current_batch_M_N_read_count * data_type.word_size

                # Calculate cycles
                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                prvious_batch_write_cycle_count = ceil(
                    current_batch_M_N_read_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )

                # Update totals
                total_read_cycle_count += current_batch_read_cycle_count
                total_write_cycle_count += prvious_batch_write_cycle_count
                total_compute_cycle_count += current_batch_compute_cycle_count


            # Calculate total cycle count
            skkim_total_cycle_count = total_read_cycle_count + total_write_cycle_count + total_compute_cycle_count
            skkim_sa_util_rate = tile_obj.util_rate

            # Debugging: Print final totals
#            print("Final Totals:")
#            print(f"Total Read Bytes: {total_read_byte}")
#            print(f"Total Write Bytes: {total_write_byte}")
#            print(f"Total Read Cycle Count: {total_read_cycle_count}")
#            print(f"Total Write Cycle Count: {total_write_cycle_count}")
#            print(f"Total Compute Cycle Count: {total_compute_cycle_count}")
#            print(f"Total Cycle Count: {skkim_total_cycle_count}")

            print(f"skkim Total Read Bytes: {current_batch_M_K_read_count}, {current_batch_K_N_read_count}, {current_batch_M_N_read_count}")
            return (
                total_read_byte,
                total_read_cycle_count,
                total_compute_cycle_count,
                total_write_byte,
                total_write_cycle_count,
                skkim_total_cycle_count,
                skkim_sa_util_rate,
            )



    class softmax_L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.flops_per_exp = (
                pcb_module.compute_module.core.vector_unit.flops_per_exp
            )
            self.read_byte = M * N * data_type.word_size
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            tmp_va_util, tmp_compute_cycle = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_cycle_count = tmp_compute_cycle
            self.va_util = tmp_va_util
            self.write_byte = M * N * data_type.word_size
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.reduction_cycle_count = (
                M
                * N
                * (self.flops_per_exp + 2)
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                + M
                * N
                * data_type.word_size
                * 2
                / (pcb_module.compute_module.l2_bandwidth_per_cycle/pcb_module.compute_module.core_count)
            )

        def calculate_softmax_vector_utilization(
                self,
                total_cycle_count: int,
                M: int,
                N: int,
                pcb_module: Device,
        ):
                # Online softmax - actual FLOP calculation
                total_flop_count = M * N * (self.flops_per_exp * 3 + 7)


                # Maximum FLOP the vector unit can perform
                max_flop = (
                        total_cycle_count
                        * pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                )

                # Calculate utilization
                utilization = total_flop_count / max_flop

                return utilization

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (pcb_module.compute_module.l2_bandwidth_per_cycle)
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            # online softmax
            total_flop_count = M * N * (self.flops_per_exp * 3 + 7)
            total_cycle_count = ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )
            utilization = self.calculate_softmax_vector_utilization(total_cycle_count, M, N, pcb_module)

#            print("SKKIM softmax",  pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle)
            return utilization, total_cycle_count


    class matmul_L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count, self.util_rate = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

            self.sram_loads, self.memory_usage = self.simulate_l1_tile_sram_usage(
                M, N, K, data_type, chiplet_module
            )
#            print("Load:",self.sram_loads, "Occu:",self.memory_usage)

        def simulate_l1_tile_sram_usage(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            chiplet_module: Device
        ):
            sram_loads = ceil((M * K + K * N + M * N) * data_type.word_size / chiplet_module.compute_module.core.SRAM_size)
            memory_usage = (M * K + K * N + M * N) * data_type.word_size
            return sram_loads, memory_usage

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
#            print(">>>>>>",M_tiling_factor ,K_tiling_factor ,N_tiling_factor)
#            print(">>>>>>",chiplet_module.compute_module.core.systolic_array_count)
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )
            tmp_compute_cycle, util_rate = FlashAttention.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
            )
#            print("SKKIM CHECK (MAC ", chiplet_module.compute_module.core.systolic_array.mac_per_cycle, "):",tmp_compute_cycle)

            compute_cycle_count = ceil(tmp_compute_cycle + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )
#            print("###########################", compute_cycle_count)

#            print("sa util[%](Y3) : ", util_rate)
#`            return compute_cycle_count
#            print("SKKIM matmul1", tmp_compute_cycle)
#            print("SKKIM matmul2", compute_cycle_count, M, N)
            return compute_cycle_count, util_rate

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
#        print(f'SKKIM start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        util_rate = -1
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
#            print(f"Utilization Rate 6: 1")
            util_rate = 1
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                ), 1
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                ),1
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98 
#                print(f"Utilization Rate 1: {util_rate:.3f}")
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                ), util_rate
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98 
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                ), util_rate

        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98 
#                print(f"Utilization Rate 3: {util_rate:.3f}")
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                ), util_rate

        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
            util_rate = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "util_rate"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
                util_rate = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "util_rate"
                ].item()
#                print(f"Utilization Rate 4: {util_rate:.3f}")
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util / 100
                '''
                ~/miniconda3/envs/llmcompass_ae/lib/python3.9/site-packages/scalesim$ vim single_layer_sim.py
                self.overall_util = (self.num_compute * 100) / (self.total_cycles * self.num_mac_unit)
                '''
#                print(f"Utilization Rate 5: {util_rate:.3f}")
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock), util_rate


