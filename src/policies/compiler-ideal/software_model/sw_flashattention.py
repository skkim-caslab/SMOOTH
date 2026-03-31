from utils import size
from typing import List, Tuple
from hardware_model.hw_device import Device
from software_model.sw_operators import Operator
from software_model.sw_utils import Tensor, DataType
import software_model.sw_sramcontroller as sram
import software_model.sw_tile as tile
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
import json
from scalesim.scale_sim import scalesim
import copy
from software_model.sw_softmax import Softmax
import math
import multiprocessing

class FlashAttention(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.Query = None
        self.Key = None
        self.Value = None
        self.W0 = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input_q: Tensor, input_k: Tensor, input_v: Tensor, input_w0: Tensor, sram_size:int, seq_len: int, dim: int, head: int) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        self.seq_len = seq_len
        self.d = dim
        self.d_h = dim // head
        # Set block sizes
        #self.B_c = math.ceil(sram_size / (4 * self.d))  # Ensure block size fits in SRAM
        self.B_c = math.ceil(sram_size / (4 * self.d))  # Ensure block size fits in SRAM
        #self.B_r = min(math.ceil(sram_size / (4 * self.d)), self.d)
        self.B_r = self.d_h

        # Divide matrices into blocks
        self.T_c = math.ceil(self.seq_len / self.B_c)  # Number of column blocks
        self.T_r = math.ceil(1 / self.B_r)  # Number of row blocks
 
        self.Query = input_q
        self.Key = input_k
        self.Value = input_v
        self.W0 = input_w0
        self.output_shape = input_q.shape
        flops = 0
        
#        self.Query = self.Query.reshape("c","abd")
#        self.Key = self.Key.reshape("abc","d")
#        self.Value = self.Value.reshape("abc","d")

        print("Check in block size:", self.B_c, self.B_r, self.T_c, self.T_r, self.seq_len)
        for j in range(self.T_c):
            for i in range(self.T_r):
                self.M = self.Query.shape[-2]
                self.N = self.Key.shape[-1]
                self.K = self.Key.shape[-2]

                # TODO: skkim - 1. call computational_graph, 2. flop_count, 3. io_count for softmax, SV, W0_projection 
#                self.computational_graph = self.ComputationalGraph(
#                    self.M, self.N, self.K, self.data_type
#                )

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

#        w0_latency = 0
        
#        print("SKKIM3", self.Query.shape, self.Key.shape, self.Value.shape)
#        self.Query = self.Query.reshape("c","abd")
#        self.Key = self.Key.reshape("abc","d")
#        self.Value = self.Value.reshape("abc","d")
#        print("SKKIM4", self.Query.shape, self.Key.shape, self.Value.shape)
        prv_write_cycle = 0
        next_read_cycle = 0

        Q_i = self.Query
        K_j_list = self.split_to_tile(self.Key.shape[-1],self.B_c)
        V_j_list = self.split_to_tile(self.Value.shape[-2],self.B_c)
        for j in range(self.T_c):
            # Load blocks K_j and V_j into SRAM
            K_j = self.Key.copy()
            K_j.shape[-1] = K_j_list[j]
            V_j = self.Value.copy()
            V_j.shape[-2] = V_j_list[j]
#            V_j = self.V.to_numpy()[j * B_c : (j + 1) * B_c, :]
#            print("K_j:", K_j)
            for i in range(self.T_r):
                # Load blocks Q_i, O_i, l_i, and m_i into SRAM
#                print("SKKIM QKV (Q,K,V)", self.Query.shape, self.Key.shape, self.Value.shape)
#                print("SKKIM QKV tile (Q,K,V)", Q_i.shape, K_j.shape, V_j.shape)
                self.M = Q_i.shape[-2]
                self.N = K_j.shape[-1]
                self.K = K_j.shape[-3] * K_j.shape[-2]
                self.computational_graph = self.ComputationalGraph(
                    self.M, self.N, self.K, self.data_type
                )
#                print("SKKIM QKT size(M,N,K)", self.M, self.N, self.K)
                tile_read_byte = self.M*self.K 

                if 'collect' in ops_name:
                    ops_name = '_collect'
                else:
                    ops_name = ''
                idx_name = str(j)+'_'+str(i)

                if 'collect' not in ops_name:
#                    print("Checkpoint 0")
                    sram.print_sram()

                if(idx_name == '0_0'):
                    unhided_io_cycle, io_cycle, compute_cycle,  tmp_qkT_latency ,util_rate = self.matmul_compile_and_simulate(
                        pcb_module, 'q_mul_k_'+idx_name+ops_name, 'None'
                    )

                else:
                    unhided_io_cycle, io_cycle, compute_cycle, tmp_qkT_latency ,util_rate = self.matmul_compile_and_simulate(
                        pcb_module, 'q_mul_k_'+idx_name+ops_name, 'q_mul_k_'+prev_idx_name+ops_name
                    )

                flash_attention_time_tick += unhided_io_cycle + compute_cycle
                print("total cycle(X) :", flash_attention_time_tick)
                print("compute cycle(X1) :", compute_cycle)
                print("io cycle(X2) :", unhided_io_cycle) #io_cycle(read+write cycle)
                print("current cycle(X3) :", unhided_io_cycle + compute_cycle) #compute_cycle + unhided_io_cycle
                print("memory bw util[%](Y1) :", 100)
                print("sram occupancy[%](Y2) :", 1 * 3 / pcb_module.compute_module.core.SRAM_size * 100)
                print("sa util[%](Y3) :", util_rate * 100)
                print("va util[%](Y3) :", 0)

                if 'collect' not in ops_name:
#                    print("Checkpoint 1")
                    sram.print_sram()

                if compute_cycle != -1: 
                    assert (compute_cycle > prv_write_cycle)
                if(idx_name == '0_0'):
                    unhided_io_cycle, io_cycle, compute_cycle, tmp_softmax_latency = self.softmax_compile_and_simulate(
                        pcb_module, 'mha_softmax' + idx_name+ops_name, 'None'
                    )
                else:
                    unhided_io_cycle, io_cycle, compute_cycle, tmp_softmax_latency = self.softmax_compile_and_simulate(
                        pcb_module, 'mha_softmax' + idx_name+ops_name, 'mha_softmax' + prev_idx_name+ops_name
                    )
#                print("SKKIM softmax size(M,N)", self.M, self.N)
                flash_attention_time_tick += compute_cycle
                print("total cycle(X) :", flash_attention_time_tick) #
                print("compute cycle(X1) :", compute_cycle)
                print("io cycle(X2) :", compute_cycle) 
#                print("unhided io cycle(X2) :", unhided_io_cycle) 
#                print("skkim io cycle(X2) :", io_cycle) 
                print("current cycle(X3) :", compute_cycle )
                print("memory bw util[%](Y1) :", 100)
                print("sram occupancy[%](Y2) :", 1 * 4 / pcb_module.compute_module.core.SRAM_size * 100)
                print("sa util[%](Y3) :", util_rate * 100)
                print("va util[%](Y3) :", 0)

#                print("SKKIM softmax:",tmp_read_byte, tmp_read_cycle, tmp_compute_cycle, tmp_write_byte, tmp_write_cycle, tmp_softmax_latency)
                #flash_attention_tot_cycle += tmp_compute_cycle
                #flash_attention_compute_cycle += tmp_compute_cycle
                #flash_attention_time_tick += tmp_compute_cycle

#                print("check sv")
                self.M = Q_i.shape[-2]
                self.N = V_j.shape[-1]
                self.K = V_j.shape[-3] * V_j.shape[-2]
                self.computational_graph = self.ComputationalGraph(
                    self.M, self.N, self.K, self.data_type
                )
                if 'collect' not in ops_name:
#                    print("Checkpoint 2")
                    sram.print_sram()

                if(idx_name == '0_0'):
                    unhided_io_cycle, io_cycle, compute_cycle, tmp_sv_latency ,util_rate = self.matmul_compile_and_simulate(
                        pcb_module, 'a_mul_v_'+idx_name+ops_name, 'None'
                    )
                else:
                    unhided_io_cycle, io_cycle, compute_cycle, tmp_sv_latency ,util_rate = self.matmul_compile_and_simulate(
                        pcb_module, 'a_mul_v_'+idx_name+ops_name, 'a_mul_v_' + prev_idx_name+ops_name
                    )

                flash_attention_time_tick += compute_cycle
#                print("SKKIM SV size(M,N,K)", self.M, self.N, self.K)
                print("total cycle(X) :", flash_attention_time_tick)
                print("compute cycle(X1) :", compute_cycle)
                print("io cycle(X2) :", compute_cycle) #io_cycle(read+write cycle)
                print("current cycle(X3) :", compute_cycle) #compute_cycle + unhided_io_cycle
                print("memory bw util[%](Y1) :", 100)
                print("sram occupancy[%](Y2) :", 1 * 3 / pcb_module.compute_module.core.SRAM_size * 100)
                print("sa util[%](Y3) :", util_rate * 100)
                print("va util[%](Y3) :", 0)

#                if 'collect' not in ops_name:
#                    print("Checkpoint 3")
#                    sram.print_sram()
                qkT_latency += tmp_qkT_latency
                softmax_latency = tmp_softmax_latency
                sv_latency += tmp_sv_latency
                prev_idx_name = idx_name
        return qkT_latency + softmax_latency + sv_latency



    def matmul_compile_and_simulate(
        self,
        pcb_module: Device,
        ops_name : str,
        prev_ops_name : str,
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
#        next_ops_name = 'w1_projection'

        if 'q_mul_k' in ops_name:
            next_ops_name = 'a_mul_v'
        elif 'a_mul_v' in ops_name:
            next_ops_name = 'w0_projection'

        l2_loop_order = "knm"
        l1_loop_order = "knm"
        l0_M_tiling_factor = 1
        l0_N_tiling_factor = 1
        l0_K_tiling_factor = 1
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
        unhided_io_cycle, io_cycle, compute_cycle, cycle_count, util_rate = self.matmul_simulate(
            self.computational_graph,
            mapping,
            pcb_module,
            ops_name,
            next_ops_name,
            prev_ops_name
        )
            # end=time.time()
            # print(f'simulation time: {end-start}')
#
        M_size = M
        N_size = N
        K_size = K
        if mapping.is_l2_double_buffering:
            occupacy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size * 2 / pcb_module.compute_module.core.SRAM_size
        else:
            occupacy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size / pcb_module.compute_module.core.SRAM_size
#        print("Tile size, ",M_size, N_size, K_size)
#        print("Word size, ",self.data_type.word_size)
#        print("SRAM size, ",pcb_module.compute_module.core.SRAM_size)
#        print("IO BW, ",pcb_module.compute_module.l2_bandwidth_per_cycle)
#        print("SRAM Util, ",occupacy)
#        print("Min Cycle, ",min_cycle_count)

        self.best_cycle_count = cycle_count

        self.unhided_io_cycle = unhided_io_cycle
        self.io_cycle = io_cycle
        self.compute_cycle = compute_cycle

        self.best_latency = cycle_count
        self.latency = self.best_latency
        self.best_mapping = mapping
#        self.best_mapping.display()
        self.util_rate = util_rate

#        print("Occupancy(%)/sram:",occupancy,"/",pcb_module.compute_module.core.SRAM_size)
        return self.unhided_io_cycle, self.io_cycle, self.compute_cycle, self.latency, self.util_rate

    def matmul_simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: matmul_Mapping,
        pcb_module: Device,
        ops_name : str,
        next_ops_name : str,
        prev_ops_name : str,
    ) -> int:
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
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
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
                prev_ops_name,
            )


        initialize_tile(0,0,0, l2_tile_M, l2_tile_N, l2_tile_K)

        # Initialize performance metrics
        total_cycle_count = 0
        skkim_unhided_io_cycle_count = 0
        skkim_io_cycle_count = 0
        skkim_compute_cycle_count = 0

        # Process tiles
        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K), mapping.l2_loop_order
        ):
            current_tile = l2_tiles[m, n, k]

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

            skkim_unhided_io_cycle_count += current_tile.total_unhided_io_cycle_count
            skkim_io_cycle_count += current_tile.total_io_cycle_count
            skkim_compute_cycle_count += current_tile.total_compute_cycle_count
            skkim_util_rate = current_tile.util_rate

        # Add final tile cycles
        final_tile = l2_tiles[-1, -1, -1]

        total_cycle_count += (
            final_tile.M_N_io_cycle_count + final_tile.compute_cycle_count
        )
        if K > l2_tile_K:
            total_cycle_count += final_tile.K_reduction_cycle_count

        # Return results
        return (
            skkim_unhided_io_cycle_count,
            skkim_io_cycle_count,
            skkim_compute_cycle_count,
            total_cycle_count,
            skkim_util_rate
        )

    def softmax_compile_and_simulate(self, pcb_module: Device, ops_name: str, prev_ops_name : str, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.K // self.d_h
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = M
        l1_tile_N = N
        l1_tile_M = M
        is_l2_double_buffering = False
        is_l1_double_buffering = False

        if is_l1_double_buffering:
            assert(
                l1_tile_M * l1_tile_N * data_type.word_size
                <= pcb_module.compute_module.core.SRAM_size // 2
            )
        else:
            assert(
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
        unhided_io_cycle, io_cycle, compute_cycle, cycle_count = self.softmax_simulate(
            self.computational_graph, mapping, pcb_module
        )
        self.best_mapping = mapping
        self.best_cycle_count = cycle_count
        self.unhided_io_cycle = unhided_io_cycle
        self.io_cycle = io_cycle
        self.compute_cycle = compute_cycle

        self.best_latency = cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
#        self.best_mapping.display()
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
#        print("Tile size, ",M_size, N_size)
        
        return self.unhided_io_cycle, self.io_cycle, self.compute_cycle, self.latency

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
        total_cycle_count = 0
        skkim_unhided_io_cycle_count = 0
        skkim_io_cycle_count = 0
        skkim_compute_cycle_count = 0

        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(ceil(M / l2_tile_M)):
            current_tile = l2_tiles[m]

            total_cycle_count += current_tile.compute_cycle_count

            skkim_io_cycle_count += current_tile.total_io_cycle_count
            skkim_compute_cycle_count += current_tile.total_compute_cycle_count

        return (
            skkim_unhided_io_cycle_count,
            skkim_io_cycle_count,
            skkim_compute_cycle_count,
            total_cycle_count,
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
            total_unhided_io_cycle_count, total_io_cycle_count, total_compute_cycle_count, skkim_total_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_cycle_count = skkim_total_cycle_count

            self.total_unhided_io_cycle_count = total_unhided_io_cycle_count
            self.total_io_cycle_count = total_io_cycle_count
            self.total_compute_cycle_count = total_compute_cycle_count
            self.skkim_total_cycle_count = skkim_total_cycle_count


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

            # Cycle count variables
            total_unhided_io_cycle_count = 0
            total_io_cycle_count = 0
            total_compute_cycle_count = 0
            skkim_total_cycle_count = 0

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
#skkim remove softmax read write
            l1_tile_compute_cycle_count = (
                l1_tile.compute_cycle_count
            )

            total_compute_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
                l1_tile_compute_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )

            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
                l1_tile_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )

            #skkim
            total_unhided_io_cycle_count = 0
            total_io_cycle_count = l1_tile.write_cycle_count + l1_tile.read_cycle_count
            total_compute_cycle_count = l1_tile_compute_cycle_count
            skkim_total_cycle_count = total_cycle_count

#            print("softmax loadable amount cycle:", l1_tile_compute_cycle_count)
#            loadable_amount = 0
            loadable_amount = l1_tile_compute_cycle_count * pcb_module.compute_module.l2_bandwidth_per_cycle / pcb_module.compute_module.core.systolic_array.input_word_size
            sram_status = sram.load_sram_status()
            while(loadable_amount != 0):
                print('loadable_amount during softmax compute : ', loadable_amount)
                loadable_amount, sram_status = sram.load_tile_to_sram(
                    sram_status, pcb_module, loadable_amount 
                )
            sram.store_sram_status(sram_status)
            print(f"sram status: {sram_status}")
 

            return (
                total_unhided_io_cycle_count,
                total_io_cycle_count,
                total_compute_cycle_count,
                skkim_total_cycle_count,
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
            prev_ops_name : str,
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
            if 'collect' in ops_name:
                total_unhided_io_cycle_count, total_io_cycle_count, total_compute_cycle_count, skkim_total_cycle_count, skkim_util_rate = self.simulate_l2_tile_compute_cycle_count_collect(
#                self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count_collect(
                    M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name
                )
            else:
                total_unhided_io_cycle_count, total_io_cycle_count, total_compute_cycle_count, skkim_total_cycle_count, skkim_util_rate = self.simulate_l2_tile_compute_cycle_count(
                    M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name, prev_ops_name
                )

            self.compute_cycle_count = skkim_total_cycle_count
            self.total_unhided_io_cycle_count = total_unhided_io_cycle_count
            self.total_io_cycle_count = total_io_cycle_count
            self.total_compute_cycle_count = total_compute_cycle_count
            self.skkim_total_cycle_count = skkim_total_cycle_count
            self.util_rate = skkim_util_rate

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

        def simulate_l2_tile_compute_cycle_count_collect(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
            ops_name : str,
        ) -> int:
            l1_tile_M, l1_tile_N, l1_tile_K = mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K
            # Cycle count variables
            total_read_byte = 0
            total_read_cycle_count = 0
            total_compute_cycle_count = 0
            total_write_byte = 0
            total_write_cycle_count = 0
            skkim_total_cycle_count = 0

            total_cycle_count = 0

            # Calculate L1 tile configurations
            M_l1_t, N_l1_t, K_l1_t = M // l1_tile_M, N // l1_tile_N, K // l1_tile_K
            M_remain, N_remain, K_remain = M % l1_tile_M, N % l1_tile_N, K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=FlashAttention.matmul_L1TileSimulator,
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

            # Active L1 tile simulation
            for m, n, k in FlashAttention.generate_tile_loops(
                ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K), mapping.l1_loop_order
            ):
                active_tiles = [(m, n, k, l1_tiles[m, n, k])]

                current_batch_M_K_read_count = 0
                current_batch_K_N_read_count = 0
                current_batch_M_N_read_count = 0
                previous_batch_M_N_write_count = 0
                previous_batch_compute_cycle_count = 0
                current_batch_compute_cycle_count = 0

                for tmp_tile in active_tiles:
                    tile_m, tile_n, tile_k, tile_obj = tmp_tile
                    current_batch_M_K_read_count += tile_obj.M * tile_obj.K
                    current_batch_K_N_read_count += tile_obj.K * tile_obj.N
                    current_batch_M_N_read_count += tile_obj.M * tile_obj.N
                    previous_batch_M_N_write_count += tile_obj.M * tile_obj.N

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
                if current_batch_M_K_read_count > 0 and 'q_mul_k' in ops_name:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 0,
                    )

                if current_batch_K_N_read_count > 0 and 'q_mul_k' in ops_name:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 1,
                    )

                if current_batch_M_N_read_count > 0 and 'q_mul_k' in ops_name:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 2,
                    )


                if (current_batch_M_K_read_count > 0 or current_batch_K_N_read_count > 0 or current_batch_M_N_read_count > 0) and 'q_mul_k' in ops_name:
                    if (m, n, k) == (0, 0, 0):
                        tile.collect_alloc_tile(
                            m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name
                        )
                    else:
                        if previous_batch_M_N_write_count > 0:
                            tile.collect_alloc_tile(
                                m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name
                            )
                

                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                prvious_batch_write_cycle_count = ceil(
                    previous_batch_M_N_write_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
#                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
#                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
#                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
#                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write
            '''
            total_cycle_count += previous_batch_compute_cycle_count + ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )
            '''

            process_id = multiprocessing.current_process().pid
            if 'w2_projection' in ops_name:
    #            file_path = "./Tiles/whole_tile_list.json"
                process_dir = f"./Tiles/"
                os.makedirs(process_dir, exist_ok=True)

                file_path = os.path.join(process_dir, f"whole_tile_list_{process_id}.json")
                with open(file_path, 'r') as f:
                    data = json.load(f)


#                remained_file_path = "./Tiles/remained_tile_list.json"
                process_dir = f"./Tiles/"
                os.makedirs(process_dir, exist_ok=True)

                file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
                with open(remained_file_path, 'w') as f:
                    json.dump(data, f, indent=2)

            return -1, -1, -1, -1, -1



        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
            ops_name : str,
            prev_ops_name : str,
        ) -> int:

            l1_tile_M, l1_tile_N, l1_tile_K = mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K
            l1_tile_M, l1_tile_N, l1_tile_K = mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K
            
            # Calculate L1 tile configurations
            M_l1_t, N_l1_t, K_l1_t = M // l1_tile_M, N // l1_tile_N, K // l1_tile_K
            M_remain, N_remain, K_remain = M % l1_tile_M, N % l1_tile_N, K % l1_tile_K

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
            total_unhided_io_cycle_count = 0
            total_io_cycle_count= 0
            total_compute_cycle_count = 0
            skkim_total_cycle_count = 0

            util_rate = l1_tiles[0,0,0].util_rate

            total_cycle_count = 0
            previous_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            )
            previous_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_compute_cycle_count = 0
            active_l1_tile_list = []

            sram_status = sram.load_sram_status()

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
                if (chiplet_module.compute_module.core.systolic_array.input_word_size != chiplet_module.compute_module.core.systolic_array.output_word_size):
                    raise Exception('Input_word_size and output_word_size is not same!!! Correcting below code is needed')


                is_loaded, needed_tile = sram.flashattention_check_needed_tile(sram_status, ops_name)
                write_or_free_ended = False
                unhided_io_amount = 0
#                if (is_loaded == True):
#                    print ("SKKIM HIT", ops_name + "_" + str(m)  + "_" + str(n)  + "_" + str(k))
#                elif (is_loaded == False):
#                    print ("SKKIM MISS", ops_name + "_" + str(m)  + "_" + str(n)  + "_" + str(k))

                while(is_loaded == False):
                    loadable_amount = chiplet_module.compute_module.core.SRAM_size
                    if(write_or_free_ended):
                        remained_amount, sram_status = sram.load_tile_to_sram(
                            sram_status, chiplet_module, loadable_amount
                        )
                        unhided_io_amount += loadable_amount - remained_amount
                    else:
                        if('q_mul_k_0_0' in ops_name):
                            loadable_amount = chiplet_module.compute_module.core.SRAM_size
                            remained_amount, sram_status = sram.write_previous_ops_from_sram(
                                sram_status, ops_name, loadable_amount
                            )
                            unhided_io_amount += loadable_amount - remained_amount
                        elif('q_mul_k' in ops_name):
                            loadable_amount = chiplet_module.compute_module.core.SRAM_size
                            remained_amount, sram_status = sram.flash_attention_write(
                                sram_status, prev_ops_name, chiplet_module, ops_name, loadable_amount
                            )
                            unhided_io_amount += loadable_amount - remained_amount
                        write_or_free_ended = True

                    is_loaded, needed_tile = sram.flashattention_check_needed_tile(sram_status, ops_name)


#                print(ops_name, "loadable amount cycle:", current_batch_compute_cycle_count)
#                if 'q_mul_k' in ops_name:
#                    loadable_amount = (current_batch_compute_cycle_count*2+208) * chiplet_module.compute_module.l2_bandwidth_per_cycle / chiplet_module.compute_module.core.systolic_array.input_word_size
#                else:
#                    loadable_amount = 0
                loadable_amount = current_batch_compute_cycle_count * chiplet_module.compute_module.l2_bandwidth_per_cycle / chiplet_module.compute_module.core.systolic_array.input_word_size
                start_sram_size = sram.get_sramutil(sram_status) 
                end_write_sram_size = start_sram_size
                hided_io_cycle_count = current_batch_compute_cycle_count
                skkim_io_byte = 0
                if('q_mul_k_0_0' in ops_name):
                    remained_amount, sram_status = sram.write_previous_ops_from_sram(
                        sram_status, ops_name, loadable_amount
                    )
                    end_write_sram_size = sram.get_sramutil(sram_status)
                    loadable_amount = remained_amount
                elif('q_mul_k' in ops_name):
                    remained_amount, sram_status= sram.flash_attention_write(
                        sram_status, prev_ops_name, chiplet_module, ops_name, loadable_amount
                    )
                    end_write_sram_size = sram.get_sramutil(sram_status)
                    skkim_io_byte += abs(end_write_sram_size - start_sram_size)
                    loadable_amount = remained_amount

                load_cnt = 0
                while(loadable_amount != 0):
                    previous_sram = sram_status.copy()
                    prev_loadable_amount = loadable_amount
                    loadable_amount, sram_status = sram.load_tile_to_sram(
                        sram_status, chiplet_module, loadable_amount 
                    )
                    time.sleep(0.1)
                    load_cnt += 1
                    if(previous_sram == sram_status):
                        hided_io_cycle_count = hided_io_cycle_count - (prev_loadable_amount * chiplet_module.compute_module.core.systolic_array.input_word_size / chiplet_module.compute_module.l2_bandwidth_per_cycle)
                if load_cnt == 1 or load_cnt == 0:
                    skkim_io_byte = 0
                else:
                    skkim_io_byte += abs(sram.get_sramutil(sram_status) - end_write_sram_size)
                unhided_io_cycle_count = unhided_io_amount * chiplet_module.compute_module.core.systolic_array.input_word_size / chiplet_module.compute_module.l2_bandwidth_per_cycle
                skkim_io_cycle = hided_io_cycle_count + unhided_io_cycle_count
                skkim_compute_cycle = current_batch_compute_cycle_count
                skkim_total_cycle_count = current_batch_compute_cycle_count + unhided_io_cycle_count


                current_total_cycle_count = (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count

                active_l1_tile_list = []
                skkim_compute_cycle =  previous_batch_compute_cycle_count
                # Update totals
                total_unhided_io_cycle_count += unhided_io_cycle_count
                total_io_cycle_count += skkim_io_cycle
                total_compute_cycle_count += current_batch_compute_cycle_count

            sram.store_sram_status(sram_status)
            skkim_total_cycle_count = total_unhided_io_cycle_count + total_compute_cycle_count
            #print('total_io_cycle_count : ', total_io_cycle_count)

            return (
                total_unhided_io_cycle_count,
                total_io_cycle_count,
                total_compute_cycle_count,
                skkim_total_cycle_count,
                util_rate,
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
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
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
            return ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )


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
            self.compute_cycle_count, self.util_rate= self.simulate_l1_tile_compute_cycle_count(
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
            compute_cycle_count = ceil(tmp_compute_cycle
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

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
        util_rate = -1
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
            uril_rate = 1
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                ), util_rate
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                ), util_rate
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98 
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
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                ), util_rate
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
            util_rate  = look_up_table.loc[
                (N, M, K, array_height, array_width, dataflow), "util_rate"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
                util_rate  = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "util_rate"
                ].item()
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
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util/100
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

