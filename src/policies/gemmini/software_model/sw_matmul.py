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
import json

class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output3_shape = None
        self.config_file = None

    def __call__(self, input1: Tensor, input2: Tensor, config_file) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
        self.bs = size(self.input1_shape[:-2])
        self.M = self.input1_shape[-2]
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.config_file = config_file
        return output

    def compile_and_simulate(self, pcb_module: Device, ops_name: str):

        matmul = Matmul(self.data_type)
        _ = matmul(
            Tensor([self.M, self.K * self.bs]), Tensor([self.K * self.bs, self.N]), self.config_file
        )
        matmul_latency1 = matmul.compile_and_simulate(pcb_module, ops_name)
        matmul_latency2 = (
            matmul_latency1
            + (self.bs - 1)
            * self.M
            * self.N
            * self.data_type.word_size
            / pcb_module.io_module.bandwidth
        )
        self.latency = matmul_latency2
        return self.latency

class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None
        self.tile_config = None

    def __call__(self, input1: Tensor, input2: Tensor, config_file) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N
        with open(config_file, 'r') as f:
            self.tile_config = json.load(f)
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

    class Mapping:
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

    def compile_and_simulate(
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
        for key in self.tile_config:
            if key in ops_name:
                l1_tile_M = self.tile_config[key]['l1_tile_M']
                l1_tile_N = self.tile_config[key]['l1_tile_N']
                l1_tile_K = self.tile_config[key]['l1_tile_K']
                next_ops_name = self.tile_config[key]['next_ops_name']
                break

        else:
            raise Exception('Not matmul case')

        cache_config = f"configs/{ops_name}_{pcb_module.compute_module.l2_bandwidth_per_cycle}_{pcb_module.compute_module.core.SRAM_size}_{self.M}_{self.N}_{self.K}_{l1_tile_M}_{l1_tile_N}_{l1_tile_K}.cfg"
        '''
        if os.path.exists(cache_config):
            with open(cache_config, 'r') as f:
                cached_data = json.load(f)
                self.best_cycle_count = cached_data.get('best_cycle_count', 0)
                self.latency = cached_data.get('latency', self.latency)
                print("total cycle:",self.best_cycle_count)
                return self.best_cycle_count
        '''

        l2_loop_order = "knm"
        l1_loop_order = "knm"
        l0_M_tiling_factor = 1
        l0_N_tiling_factor = 1
        l0_K_tiling_factor = 1
        mapping = self.Mapping(
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
        cycle_count = self.simulate(
            self.computational_graph,
            mapping,
            pcb_module,
            ops_name,
            next_ops_name
        )

        self.best_mapping = mapping
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
        K_size = self.best_mapping.l1_tile_K
        if mapping.is_l2_double_buffering:
            occupancy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size * 2 / pcb_module.compute_module.core.SRAM_size * 100
        else:
            occupancy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size / pcb_module.compute_module.core.SRAM_size * 100
        self.best_cycle_count = cycle_count
        self.best_latency = cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        self.best_mapping.display()
        cached_data = {
            'best_cycle_count': self.best_cycle_count,
            'latency': self.latency,
        }
        with open(cache_config, 'w') as f:
            json.dump(cached_data, f, indent=4)

        return self.best_cycle_count

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        ops_name : str,
        next_ops_name : str,
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
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            )
        else:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
#        print('-'*20)
#        print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                ops_name,
                next_ops_name,
            )

        # compute and write last tile
        return l2_tiles[-1, -1, -1].compute_cycle_count
        

    class L2TileSimulator:
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
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name
            )

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
                l1_tiles[m, n, k] = Matmul.L1TileSimulator(
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

            util_rate = l1_tiles[0,0,0].util_rate
            M_K_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
            )
            M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain

            K_N_tile_size = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
            )
            K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain

            M_N_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
            )
            M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain


            total_cycle_count = 0
            io_total_cycle_count = 0
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

            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ):
                active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
                if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
                ):
                    pass
                elif (
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                ):
                    continue

                assert (
                    len(active_l1_tile_list) <= chiplet_module.compute_module.core_count
                )
                current_batch_Read_M_K = np.zeros(
                    [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                )
                current_batch_Read_K_N = np.zeros(
                    [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Read_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Write_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )

                current_batch_compute_cycle_count = 0
                for i in range(len(active_l1_tile_list)):
                    temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                    current_batch_Read_M_K[temp_m, temp_k] = 1
                    current_batch_Read_K_N[temp_k, temp_n] = 1
                    current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                    current_batch_Write_M_N[temp_m, temp_n] = 1
                    temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                    if temp_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            temp_l1_tile.M
                            * temp_l1_tile.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count,
                        temp_l1_tile_compute_cycle_count,
                    )

                # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
                # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
                current_batch_M_K_read_count = np.sum(
                    (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                    * M_K_tile_size
                )

                current_batch_K_N_read_count = np.sum(
                    (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                    * K_N_tile_size
                )
                current_batch_M_N_read_count = np.sum(
                    (
                        current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                    )
                    * M_N_tile_size
                )
                previous_batch_M_N_write_count = np.sum(
                    (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                    * M_N_tile_size
                )

                # read current batch while compute and write previous batch
                current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
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
                current_cycle_count = (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )
                print("total cycle(X) : ",total_cycle_count)
                print("compute cycle(X1) : ",previous_batch_compute_cycle_count)
                print("SA cycle(X1) : ",previous_batch_compute_cycle_count)
                print("io cycle(X2) : ",(current_batch_read_cycle_count + prvious_batch_write_cycle_count))
                io_total_cycle_count += (max(current_batch_read_cycle_count-previous_batch_compute_cycle_count, 0) + prvious_batch_write_cycle_count)
                print("accum io cycle(X2) : ", io_total_cycle_count)
                print("current cycle(X3) : ",current_cycle_count)
                loadable_amount = current_cycle_count-(current_batch_read_cycle_count + prvious_batch_write_cycle_count)
                if loadable_amount != 0:
                    print("loadable cycle(X4) : ", loadable_amount)

                print("memory bw util[%](Y1) : ", (current_batch_read_cycle_count + prvious_batch_write_cycle_count)/current_cycle_count * 100) 
                print("sram occupancy[%](Y2) : ", (l1_tile_M * l1_tile_N + l1_tile_N * l1_tile_K + l1_tile_M * l1_tile_K)*2/chiplet_module.compute_module.core.SRAM_size * 100)
                print("sa util[%](Y3) : ", util_rate * 100)
                print("va util[%](Y3) : ", 0)



                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write
            total_cycle_count += previous_batch_compute_cycle_count + ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )

            last_write_cycle_count = ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )

            print("total cycle(X) : ",total_cycle_count)
            print("compute cycle(X1) : ",previous_batch_compute_cycle_count)
            print("SA cycle(X1) : ",previous_batch_compute_cycle_count)
            print("io cycle(X2) : ", last_write_cycle_count)
            print("current cycle(X3) : ", previous_batch_compute_cycle_count + last_write_cycle_count)
            print("memory bw util[%](Y1) : ", last_write_cycle_count/(previous_batch_compute_cycle_count + last_write_cycle_count)*100)
            print("sram occupancy[%](Y2) : ", (current_batch_read_count * chiplet_module.compute_module.core.systolic_array.input_word_size + previous_batch_M_N_write_count* chiplet_module.compute_module.core.systolic_array.output_word_size)/ chiplet_module.compute_module.core.SRAM_size * 100)
            print("sa util[%](Y3) : ", util_rate * 100)
            print("va util[%](Y3) : ", 0)


            return total_cycle_count
            


    class L1TileSimulator:
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
            required_space = M * K + K * N + M * N
            available_space = chiplet_module.compute_module.core.SRAM_size // data_type.word_size // 2
            assert required_space <= available_space, (
               f"SRAM capacity exceeded: required={required_space}, available={available_space} "
               f"M={M}, N={N}, K={K}"
               f"({required_space / 1024:.2f} KB required, {available_space * data_type.word_size / 1024:.2f} KB available)"
            )            


            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )
            tmp_compute_cycle, util_rate = Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
            )

            compute_cycle_count = ceil(tmp_compute_cycle + (K_tiling_factor - 1)
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
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
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

        key = (M,N,K,array_height, array_width, dataflow)
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
            except KeyError:
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
        return ceil(cycle_count / mac_per_clock), util_rate


