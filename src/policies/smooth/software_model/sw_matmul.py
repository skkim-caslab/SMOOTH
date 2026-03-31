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
import multiprocessing
import math
process_id = multiprocessing.current_process().pid

class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output3_shape = None

    def __call__(self, input1: Tensor, input2: Tensor, config_file) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
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
        """
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency1 = (
            matmul.compile_and_simulate(pcb_module, compile_mode) * self.bs
        )
        """

        matmul = Matmul(self.data_type)
        _ = matmul(
            Tensor([self.M, self.K * self.bs]), Tensor([self.K * self.bs, self.N]), self.config_file
        )
        matmul_latency2 = (
            matmul.compile_and_simulate(pcb_module, ops_name)
            + (self.bs - 1)
            * self.M
            * self.N
            * self.data_type.word_size
            / pcb_module.io_module.bandwidth
        )
        self.latency = matmul_latency2
        #self.latency = min(matmul_latency1, matmul_latency2)
        #print(f"Batchtest, {self.latency} , {matmul_latency1}, {matmul_latency2}")
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
        # print(f'{self.M}, {self.N}, {self.K}')
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
        print("DEBUG", ops_name, M,N,K)

        l2_tile_M = self.computational_graph.M
        l2_tile_N = self.computational_graph.N
        l2_tile_K = self.computational_graph.K

        is_l2_double_buffering = True
        for key in self.tile_config:
            if key in ops_name:
                l1_tile_M = self.tile_config[key]['l1_tile_M']
                l1_tile_N = self.tile_config[key]['l1_tile_N']
                l1_tile_K = self.tile_config[key]['l1_tile_K']
                next_ops_name = self.tile_config[key]['next_ops_name']
                break
        else:
            raise Exception('Not matmul case')


        l2_loop_order = "knm"
        l1_loop_order = "knm"
        for (
            l0_M_tiling_factor,
            l0_N_tiling_factor,
            l0_K_tiling_factor,
        ) in [(1, 1, 1)]:
#                    ) in [(1, 2, 1)]: #skkim
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
#                        mapping.display()
            # start=time.time()
            cycle_count = self.simulate(
                self.computational_graph,
                mapping,
                pcb_module,
                ops_name,
                next_ops_name
            )
            # end=time.time()
            # print(f'simulation time: {end-start}')
            if cycle_count < min_cycle_count:
                min_cycle_count = cycle_count
                best_mapping = mapping

        self.best_mapping = best_mapping
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
        K_size = self.best_mapping.l1_tile_K
        if mapping.is_l2_double_buffering:
            occupacy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size * 2 / pcb_module.compute_module.core.SRAM_size
        else:
            occupacy = (M_size*N_size + M_size*K_size + K_size*N_size) * self.data_type.word_size / pcb_module.compute_module.core.SRAM_size
#        print("CCCHECK",occupacy,M_size, N_size,K_size,self.data_type.word_size,pcb_module.compute_module.core.SRAM_size)
        print("Tile size, ",M_size, N_size, K_size)
        print("Word size, ",self.data_type.word_size)
        print("SRAM size, ",pcb_module.compute_module.core.SRAM_size)
        print("IO BW, ",pcb_module.compute_module.l2_bandwidth_per_cycle)
        print("SRAM Util, ",occupacy)
        print("Min Cycle, ",min_cycle_count)
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        #self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.best_latency = min_cycle_count
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

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
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
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

        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0].M_K_io_cycle_count + l2_tiles[0, 0, 0].K_N_io_cycle_count
        )
        #print('total_cycle_count')
        #print(total_cycle_count)

        previous_m = 0
        previous_n = 0
        previous_k = 0

        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            l2_tile = l2_tiles[m, n, k]
            previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]

            # current tile read latency
            if m == previous_m and k == previous_k:
                current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
            elif n == previous_n and k == previous_k:
                current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
            else:
                current_tile_read_cycle_count = (
                    l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
                )
            if k > 0 and not (m == previous_m and n == previous_n):
                current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if k > 0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if m == previous_m and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count

            # read current tile, compute previous tile, write previous tile
            if mapping.is_l2_double_buffering:  # pipelined
                total_cycle_count += (
                    max(
                        current_tile_read_cycle_count, previous_tile_compute_cycle_count
                    )
                    + previous_tile_write_cycle_count
                )
            else:  # non-pipelined
                total_cycle_count += (
                    current_tile_read_cycle_count
                    + previous_tile_compute_cycle_count
                    + previous_tile_write_cycle_count
                )

            previous_m = m
            previous_n = n
            previous_k = k

        # compute and write last tile
        total_cycle_count += (
            l2_tiles[-1, -1, -1].M_N_io_cycle_count
            + l2_tiles[-1, -1, -1].compute_cycle_count
        )

        if previous_k > 0:
            total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)

        return total_cycle_count #+ ceil(
        # pcb_module.io_module.latency * 2 * pcb_module.compute_module.clock_freq
        # )

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
            if 'collect' in ops_name:
                self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count_collect(
                    M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name
                )
            else:
                self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                    M, N, K, data_type, mapping, pcb_module, look_up_table, ops_name
                )


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
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )

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

            total_iterations = math.ceil(M / l1_tile_M) * math.ceil(N / l1_tile_N) * math.ceil(K / l1_tile_K)
            log_interval = max(1, total_iterations // 10) 
            print(M,N,K, l1_tile_M,l1_tile_N,l1_tile_K)
            for count, (m, n, k) in enumerate(Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            )):
#                if count % log_interval == 0 or count == total_iterations - 1:
#                    print(f"Progress: {count + 1}/{total_iterations} iterations completed")
                #print(m, n, k)
                #print(ceil(M / l1_tile_M), ceil(N/ l1_tile_N), ceil(K / l1_tile_K))
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
                #print('active_l1_tile_list :', active_l1_tile_list)

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
#                print("DEBUG NK", current_batch_K_N_read_count,current_batch_Read_K_N , (~previous_batch_Read_K_N),K_N_tile_size)
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
                if current_batch_M_K_read_count > 0:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 0, current_batch_M_K_read_count, 
                    )

                if current_batch_K_N_read_count > 0:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 1,current_batch_K_N_read_count,
                    )

                if current_batch_M_N_read_count > 0:
                    tile.collect_tile(
                        m, n, k, l1_tile_M, l1_tile_N, l1_tile_K, chiplet_module, ops_name, 2, current_batch_M_N_read_count 
                    )


                if current_batch_M_K_read_count > 0 or current_batch_K_N_read_count > 0 or current_batch_M_N_read_count > 0:
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

            #skkim test
            if 'w2_projection' in ops_name:
                file_path = f"./Tiles/whole_tile_list_{process_id}.json"
                with open(file_path, 'r') as f:
                    data = json.load(f)


                remained_file_path = f"./Tiles/remained_tile_list_{process_id}.json"
                with open(remained_file_path, 'w') as f:
                    json.dump(data, f, indent=2)

            return total_cycle_count



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
        ) -> int:
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
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

            sram_status, sram_table = sram.load_sram_status(chiplet_module)
#            print("MATMUL", sram_table)
         

            time_tick = 0
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ):
                #print(m, n, k)
                #print(ceil(M / l1_tile_M), ceil(N/ l1_tile_N), ceil(K / l1_tile_K))
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
#                print('active_l1_tile_list :', active_l1_tile_list)

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
#                print("CHECK MK ", current_batch_M_K_read_count)
#                print("CHECK KN ", current_batch_K_N_read_count)
#                print("CHECK MN ", current_batch_M_N_read_count)

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
                if (chiplet_module.compute_module.core.systolic_array.input_word_size != chiplet_module.compute_module.core.systolic_array.output_word_size):
                    raise Exception('Input_word_size and output_word_size is not same!!! Correcting below code is needed')


                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, m, n, k, ops_name)
                write_or_free_ended = False
                unhided_io_amount = 0
#                print("skkim | ops : ", ops_name + "_" + str(m)  + "_" + str(n)  + "_" + str(k))
                if (is_loaded == True):
                    print ("SKKIM HIT", ops_name + "_" + str(m)  + "_" + str(n)  + "_" + str(k))
                elif (is_loaded == False):
                    print ("SKKIM MISS", ops_name + "_" + str(m)  + "_" + str(n)  + "_" + str(k))
                while(is_loaded == False):
                    #print("SKKIM sram status", sram_status)
                    loadable_amount = chiplet_module.compute_module.core.SRAM_size
                    remained_amount = loadable_amount
                    tot_find_overhead = 0
                    if(write_or_free_ended): 
                        remained_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                            sram_status, sram_table, chiplet_module, loadable_amount 
                        )
                    else: # 만약 compute에 필요한 tile이 sram에 없다면, 우선, write를 진행한 후에 load 해야함.
                        if (m, n, k) == (0, 0, 0):
                            remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                                sram_status, sram_table, ops_name, loadable_amount, chiplet_module
                            )
                        else:
                            #if prvious_batch_write_cycle_count > 0:
                            if previous_batch_M_N_write_count > 0:
                                #print("prv batch write cycle count")
                                if current_batch_M_K_read_count > 0:
                                    remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 0, loadable_amount
                                    )
                                if current_batch_K_N_read_count > 0:
                                    remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 1, loadable_amount
                                    )
                                if current_batch_M_N_read_count > 0:
                                    remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 2, loadable_amount
                                    )
                            else:
                                #print("no prv batch write cycle count")
                                if current_batch_M_K_read_count > 0:
                                    sram_status, sram_table = sram.free_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 0,
                                    )
                                if current_batch_K_N_read_count > 0:
                                    sram_status, sram_table = sram.free_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 1,
                                    )
                                if current_batch_M_N_read_count > 0:
                                    sram_status, sram_table = sram.free_tile_from_sram(
                                        sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 2,
                                    )
                        write_or_free_ended = True
                    #unhided_io_amount += loadable_amount - remained_amount + tot_find_overhead
                    unhided_io_amount += loadable_amount - remained_amount
#                    print("DEBUG", unhided_io_amount, loadable_amount , remained_amount)
                        
                    is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, m, n, k, ops_name)


                loadable_amount = current_batch_compute_cycle_count * chiplet_module.compute_module.l2_bandwidth_per_cycle / chiplet_module.compute_module.core.systolic_array.input_word_size
                start_sram_size = sram.get_sramutil(sram_status, chiplet_module) 
                end_write_sram_size = start_sram_size
                hided_io_cycle_count = current_batch_compute_cycle_count
                skkim_io_byte = 0
                if (m, n, k) == (0, 0, 0):
                    remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                        sram_status, sram_table, ops_name, loadable_amount, chiplet_module
                    )
                    loadable_amount = remained_amount
                    end_write_sram_size = sram.get_sramutil(sram_status, chiplet_module) 
                else:
                    #if prvious_batch_write_cycle_count > 0:
                    if previous_batch_M_N_write_count > 0:
                        if current_batch_M_K_read_count > 0:
                            remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 0, loadable_amount
                            )
                        if current_batch_K_N_read_count > 0:
                            remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 1, loadable_amount
                            )
                        if current_batch_M_N_read_count > 0:
                            remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 2, loadable_amount
                            )
                    else:
                        if current_batch_M_K_read_count > 0:
                            sram_status, sram_table = sram.free_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 0,
                            )
                        if current_batch_K_N_read_count > 0:
                            sram_status, sram_table = sram.free_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 1,
                            )
                        if current_batch_M_N_read_count > 0:
                            sram_status, sram_table = sram.free_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, chiplet_module, ops_name, 2,
                            )
                        remained_amount = loadable_amount
                    end_write_sram_size = sram.get_sramutil(sram_status, chiplet_module) 
                    skkim_io_byte += abs(end_write_sram_size - start_sram_size)

                loadable_amount = remained_amount
                load_cnt = 0
                if(loadable_amount < 0):
                    unhided_io_amount += loadable_amount
                    hided_io_cycle_count = current_batch_compute_cycle_count * chiplet_module.compute_module.l2_bandwidth_per_cycle / chiplet_module.compute_module.core.systolic_array.input_word_size

                tot_find_overhead = 0
#                print(f"sram status: {sram_status}i")
#                print(f"loadable amount {m}_{n}_{k}: {loadable_amount}")
                print()
                while(loadable_amount > 0):
                    previous_sram = sram_status.copy()
                    prev_loadable_amount = loadable_amount
                    #print("SKKIM second while loop load tile")
                    tmp_sram_status = sram_status.copy()
                    #loadable_amount, sram_status, sram_table, tmp_find_overhead = sram.load_tile_to_sram_cont( #cache
                    loadable_amount, sram_status, sram_table, tmp_find_overhead = sram.load_tile_to_sram(
                        sram_status, sram_table, chiplet_module, loadable_amount 
                    )
                    tot_find_overhead += tmp_find_overhead
#                    for i in tmp_sram_status:
#                        if i not in sram_status:
#                            print("-----", i)
#                    for i in sram_status:
#                        if i not in tmp_sram_status:
#                            print("skkim | +++++(preload)", i)
#                    print("skkim | sram table", sram_table)
#                    print("skkim | sram status", sram_status)
                    load_cnt += 1
                    if(previous_sram == sram_status and prev_loadable_amount > 0):
                        if prev_loadable_amount == float("inf"):
                            hided_io_cycle_count = hided_io_cycle_count
                        else:
                            hided_io_cycle_count = max(hided_io_cycle_count - (prev_loadable_amount * chiplet_module.compute_module.core.systolic_array.input_word_size / chiplet_module.compute_module.l2_bandwidth_per_cycle),0)
#                print(f"END PRELOAD sram status: {sram_status}")
                if load_cnt == 1 or load_cnt == 0:
                    skkim_io_byte = 0
                else:
                    skkim_io_byte += abs(sram.get_sramutil(sram_status, chiplet_module) - end_write_sram_size)

                unhided_io_cycle_count = ceil(unhided_io_amount * chiplet_module.compute_module.core.systolic_array.input_word_size / chiplet_module.compute_module.l2_bandwidth_per_cycle) 
                #unhided_io_cycle_count = ceil(unhided_io_amount * chiplet_module.compute_module.core.systolic_array.input_word_size / chiplet_module.compute_module.l2_bandwidth_per_cycle) + tot_find_overhead
#                print(f"kh,compute_cycle for {ops_name} M_N_K {m}_{n}_{k} : {current_batch_compute_cycle_count}")
#                print(f"kh,io_cycle for {ops_name} M_N_K {m}_{n}_{k} : {hided_io_cycle_count + unhided_io_cycle_count}")
#                print(f"kh,total_cycle for {ops_name} M_N_K {m}_{n}_{k} : {unhided_io_cycle_count + current_batch_compute_cycle_count}")
                skkim_io_cycle = hided_io_cycle_count + unhided_io_cycle_count
#<<<<<<< HEAD
#                skkim_total_cycle_count = current_batch_compute_cycle_count + unhided_io_cycle_count
#=======
                skkim_compute_cycle = current_batch_compute_cycle_count
                print("DEBUG", hided_io_cycle_count, current_batch_compute_cycle_count,unhided_io_cycle_count)
                skkim_total_cycle_count = max(hided_io_cycle_count, current_batch_compute_cycle_count) + unhided_io_cycle_count
#>>>>>>> origin/skkim_preload
                time_tick += skkim_total_cycle_count



                current_total_cycle_count = (
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
#                skkim_io_cycle = current_batch_read_cycle_count + prvious_batch_write_cycle_count 
                print("total cycle(X) :", time_tick)
                print("compute cycle(X1) :", current_batch_compute_cycle_count)
                print("SA cycle(X1) :", current_batch_compute_cycle_count)
                print("io cycle(X2) :", skkim_io_cycle)
#                print("hided io cycle(X2) :", hided_io_cycle_count)
#                print("unhided io cycle(X2) :", unhided_io_cycle_count)
                print("current cycle(X3) :", skkim_total_cycle_count)
                print("memory bw util[%](Y1) :", 100)

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)
                previous_m_n_k = '_' + str(m) + '_' + str(n) + '_' + str(k)

                active_l1_tile_list = []
#                if skkim_io_cycle == 0:
#                    print("memory bw util[%](Y1) :", 0)
#                else:
#                print(f"sram occupancy[%](Y2) : {sram.get_sramutil(sram_status, chiplet_module)/ chiplet_module.compute_module.core.SRAM_size * 100:.3f}")
                print(f"sram status: {sram_status}")
#                print(f"sram table: {sram_table}")
                print("sa util[%](Y3) : ", util_rate * 100)

            # last batch's compute and write
#            total_cycle_count += previous_batch_compute_cycle_count + ceil(
#                np.sum(previous_batch_Write_M_N * M_N_tile_size)
#                * data_type.word_size
#                / chiplet_module.compute_module.l2_bandwidth_per_cycle
#            )
            sram.store_sram_status(sram_status, sram_table)

            return time_tick


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

