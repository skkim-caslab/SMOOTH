from utils import size
from typing import List, Tuple
from hardware_model.hw_device import Device
from software_model.sw_operators import Operator
from software_model.sw_utils import Tensor, DataType
import software_model.sw_sramcontroller as sram
import software_model.sw_tile as tile
from math import ceil, log2, log
import time
import statistics
import numpy as np
import torch
import os


@torch.compile
def layernorm_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.layer_norm(input, [input.shape[-1]])


class LayerNorm(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        return input

    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 2
        self.flop_count = self.M * self.N * 7
        self.roofline_latency = max(
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
            self.flop_count / pcb_module.compute_module.total_vector_flops,
        )
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l1_tile_M: int,
            l1_tile_N: int,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}"
            )

    def compile_and_simulate(self, pcb_module: Device, ops_name):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size) // 2
        )
        l2_tile_M = min(l2_tile_M, M)
        l1_tile_N = N
        l1_tile_M = pcb_module.compute_module.core.SRAM_size // (
            2 * l1_tile_N * data_type.word_size
        )
        l1_tile_M = min(l1_tile_M, M)
        mapping = self.Mapping(
            l2_tile_M,
            l2_tile_N,
            l1_tile_M,
            l1_tile_N,
        )
        cycle_count = self.simulate(self.computational_graph, mapping, pcb_module, ops_name)
        if cycle_count < min_cycle_count:
            min_cycle_count = cycle_count
            best_mapping = mapping
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count
        self.latency = self.best_latency
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        ops_name: str,
    ) -> int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
                ops_name
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
                ops_name
            )

        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):
            total_cycle_count += l2_tiles[m].read_cycle_count
            total_cycle_count += l2_tiles[m].compute_cycle_count
            total_cycle_count += l2_tiles[m].write_cycle_count
        return total_cycle_count

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
            ops_name: str,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            if 'collect' in ops_name:
                self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count_collect(
                    M, N, data_type, mapping, pcb_module, ops_name
                )

            else:
                self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                    M, N, data_type, mapping, pcb_module, ops_name
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

        def simulate_l2_tile_compute_cycle_count_collect(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
            ops_name: str,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = LayerNorm.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count * 3
                + l1_tile.write_cycle_count
                #+ l1_tile.compute_cycle_count
                + l1_tile.compute_mean_cycle_count
                + l1_tile.compute_var_cycle_count
                + l1_tile.compute_norm_cycle_count
            )
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count)
            ) * (
                l1_tile_cycle_count
                + (ceil(N / l1_tile_N) - 1) * (l1_tile.reduction_cycle_count)
            )
            
            file_path = "./Tiles/whole_tile_list.json"
            if os.path.exists(file_path) and 'MHA' in ops_name:
                os.remove(file_path)

            file_path = "./SRAM/sram_status.json"
            if os.path.exists(file_path) and 'MHA' in ops_name:
                os.remove(file_path)

            file_path = "./SRAM/sram_table.json"
            if os.path.exists(file_path) and 'MHA' in ops_name:
                os.remove(file_path)

            tile.collect_tile(
                M, N, -1, l1_tile_M, l1_tile_N, -1, pcb_module, ops_name[:3] + '_mean_collect', -1
            )
            tile.collect_alloc_tile(
                M, N, -1, l1_tile_M, l1_tile_N, -1, pcb_module, ops_name
            )
            tile.collect_tile(
                M, N, -1, l1_tile_M, l1_tile_N, -1, pcb_module, ops_name[:3] + '_var_collect', -1
            )
            tile.collect_tile(
                M, N, -1, l1_tile_M, l1_tile_N, -1, pcb_module, ops_name[:3] + '_norm_collect', -1
            )


            return total_cycle_count

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
            ops_name: str,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = LayerNorm.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count * 3
                + l1_tile.write_cycle_count
                + l1_tile.compute_mean_cycle_count
                + l1_tile.compute_var_cycle_count
                + l1_tile.compute_norm_cycle_count
            )
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count)
            ) * (
                l1_tile_cycle_count
                + (ceil(N / l1_tile_N) - 1) * (l1_tile.reduction_cycle_count)
            )

            sram_status, sram_table = sram.load_sram_status(pcb_module)
            # 3 reads and loads

            is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_mean')
            write_or_free_ended = False

            sram_usage = M * N * data_type.word_size * 3 + M * N * data_type.word_size
            time_tick = 0


            while(is_loaded == False):
                loadable_amount = pcb_module.compute_module.core.SRAM_size
                if(write_or_free_ended):
                    remained_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                        sram_status, sram_table, pcb_module, loadable_amount
                    )
                    time_tick += l1_tile.read_cycle_count
                    print("total cycle(X) : ",  time_tick)
                    print("compute cycle(X1) : ", 0)
                    print("io cycle(X2) : ", l1_tile.read_cycle_count)
                    print("current cycle(X3) : ",  l1_tile.read_cycle_count)
                    print('memory bw util[%](Y1) : ',  100)
                    print("va util[%](Y3) : ", 0)
                    print("sa util[%](Y2) : ",  0)

                else:
                    remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                        sram_status, sram_table, ops_name, loadable_amount, pcb_module
                    )
                    write_or_free_ended = True

                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_mean')

            loadable_amount = l1_tile.compute_mean_cycle_count * pcb_module.compute_module.l2_bandwidth_per_cycle / data_type.word_size
            remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                sram_status, sram_table, ops_name, loadable_amount, pcb_module
            )
            loadable_amount = remained_amount
            while(loadable_amount != 0): # Cycle with compute_mean cycle
                loadable_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                    sram_status, sram_table, pcb_module, loadable_amount 
                )
            time_tick += l1_tile.compute_mean_cycle_count
            print("total cycle(X) : ",  time_tick)
            print("compute cycle(X1) : ", l1_tile.compute_mean_cycle_count)
            print("VE cycle(X1) : ", l1_tile.compute_mean_cycle_count)
            print("io cycle(X2) : ", l1_tile.compute_mean_cycle_count)
            print("current cycle(X3) : ",  l1_tile.compute_mean_cycle_count)
            print("va util[%](Y3) : ", l1_tile.util_rate_var*100)
            print("sa util[%](Y2) : ",  0)
            print('memory bw util[%](Y1) : ',  100)

            is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_var')
            write_or_free_ended = False
            while(is_loaded == False):
                loadable_amount = pcb_module.compute_module.core.SRAM_size
                if(write_or_free_ended):
                    remained_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(

                        sram_status, sram_table, pcb_module, loadable_amount
                    )
                    time_tick += l1_tile.read_cycle_count
                    print("total cycle(X) : ",  time_tick)
                    print("compute cycle(X1) : ", 0)
                    print("io cycle(X2) : ", l1_tile.read_cycle_count)
                    print("current cycle(X3) : ",  l1_tile.read_cycle_count)
                    print("va util[%](Y3) : ", 0)
                    print("sa util[%](Y2) : ",  0)
                    print('memory bw util[%](Y1) : ',  100)

                else:
                    remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                        sram_status, sram_table, ops_name, loadable_amount, pcb_module
                    )
                    write_or_free_ended = True
                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_var')


            loadable_amount = l1_tile.compute_var_cycle_count * pcb_module.compute_module.l2_bandwidth_per_cycle / data_type.word_size
            while(loadable_amount != 0): # Cycle with compute_var cycle
                loadable_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                    sram_status, sram_table, pcb_module, loadable_amount 
                )
            time_tick += l1_tile.compute_var_cycle_count
            print("total cycle(X) : ",  time_tick)
            print("compute cycle(X1) : ", l1_tile.compute_var_cycle_count)
            print("VE cycle(X1) : ", l1_tile.compute_var_cycle_count)
            print("io cycle(X2) : ", l1_tile.compute_var_cycle_count)
            print("current cycle(X3) : ",  l1_tile.compute_var_cycle_count)
            print("va util[%](Y3) : ", l1_tile.util_rate_var*100)
            print("sa util[%](Y2) : ",  0)
            print('memory bw util[%](Y1) : ',  100)

            is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_norm')
            write_or_free_ended = False
            while(is_loaded == False):
                loadable_amount = pcb_module.compute_module.core.SRAM_size
                if(write_or_free_ended):
                    remained_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                        sram_status, sram_table, pcb_module, loadable_amount
                    )

                else:
                    remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                        sram_status, sram_table, ops_name, loadable_amount, pcb_module
                    )
                    write_or_free_ended = True
                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name+'_norm')
            time_tick += l1_tile.read_cycle_count
            print("total cycle(X) : ",  time_tick)
            print("compute cycle(X1) : ", 0)
            print("io cycle(X2) : ", l1_tile.read_cycle_count)
            print("current cycle(X3) : ",  l1_tile.read_cycle_count)
            print("va util[%](Y3) : ", 0)
            print("sa util[%](Y2) : ",  0)
            print('memory bw util[%](Y1) : ',  100)

            loadable_amount = l1_tile.compute_norm_cycle_count * pcb_module.compute_module.l2_bandwidth_per_cycle / data_type.word_size
            while(loadable_amount != 0): # Cycle with compute_norm cycle
                loadable_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                    sram_status, sram_table, pcb_module, loadable_amount 
                )
            time_tick += l1_tile.compute_norm_cycle_count
            print("total cycle(X) : ",  time_tick)
            print("compute cycle(X1) : ", l1_tile.compute_norm_cycle_count)
            print("io cycle(X2) : ", l1_tile.compute_norm_cycle_count)
            print("current cycle(X3) : ",  l1_tile.compute_norm_cycle_count)
            print("va util[%](Y3) : ", l1_tile.util_rate_norm*100)
            print("sa util[%](Y2) : ",  0)
            print('memory bw util[%](Y1) : ',  100)

            # 1 write
            sram.store_sram_status(sram_status, sram_table)

            time_tick += l1_tile.write_cycle_count
            print("total cycle(X) : ",  time_tick)
            print("compute cycle(X1) : ", 0)
            print("io cycle(X2) : ", l1_tile.write_cycle_count)
            print("current cycle(X3) : ",  l1_tile.write_cycle_count)
            print("va util[%](Y3) : ", 0)
            print("sa util[%](Y2) : ",  0)
            print('memory bw util[%](Y1) : ',  100)

            return time_tick

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count, self.util_rate_com = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_mean_cycle_count, self.util_rate_mean = self.simulate_l1_tile_compute_mean_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_var_cycle_count, self.util_rate_var = self.simulate_l1_tile_compute_var_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.compute_norm_cycle_count, self.util_rate_norm = self.simulate_l1_tile_compute_norm_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.reduction_cycle_count = (
                M
                * N
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                + M
                * N
                * data_type.word_size
                * 2
                / (
                    pcb_module.compute_module.l2_bandwidth_per_cycle
                    / pcb_module.compute_module.core_count
                )
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
        def calculate_vector_unit_utilization(
                self,
                total_cycle_count: int,
                M: int,
                N: int,
                pcb_module: Device,
        ):
                # Vector unit parameters
                vector_width = pcb_module.compute_module.core.vector_unit.vector_width
                vector_count = pcb_module.compute_module.core.vector_unit.vector_count
                flops_per_cycle = pcb_module.compute_module.core.vector_unit.flops_per_cycle

                # Calculate actual operations performed (Actual FLOP)
                actual_flop = (
                        M
                        * N
                        * 2  # Adjust weight depending on operation type (e.g., multiplication + addition = 2 FLOPs)
                )

                # Calculate maximum possible operations (Maximum FLOP)
                maximum_flop = (
                        total_cycle_count
                        * vector_width
                        * vector_count
                        * flops_per_cycle
                )

                # Calculate utilization
                utilization = actual_flop / maximum_flop

                return utilization

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )

            # each lane computes it own mean
            total_cycle_count = ceil(
                N_per_vector_lane
                * M_per_vector_lane
                / pcb_module.compute_module.core.vector_unit.flops_per_cycle
            )
            # the whole vector reduce to one mean
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # each lane computes it own variance
            total_cycle_count += (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 2
            )
            # the whole vector reduce to one variance
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # calculate normalized output
            total_cycle_count += (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 4
            )  # division is heavy
            utilization = self.calculate_vector_unit_utilization(total_cycle_count, M, N, pcb_module)
            return total_cycle_count, utilization

        def simulate_l1_tile_compute_mean_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )

            # each lane computes it own mean
            total_cycle_count = ceil(
                N_per_vector_lane
                * M_per_vector_lane
                / pcb_module.compute_module.core.vector_unit.flops_per_cycle
            )
            # the whole vector reduce to one mean
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            utilization = self.calculate_vector_unit_utilization(total_cycle_count, M, N, pcb_module)
            return total_cycle_count, utilization


        def simulate_l1_tile_compute_var_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )
            # each lane computes it own variance
            total_cycle_count = (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 2
            )
            # the whole vector reduce to one variance
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )

            utilization = self.calculate_vector_unit_utilization(total_cycle_count, M, N, pcb_module)
            return total_cycle_count, utilization

        def simulate_l1_tile_compute_norm_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )
            # calculate normalized output
            total_cycle_count = (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 4
            )  # division is heavy

            utilization = self.calculate_vector_unit_utilization(total_cycle_count, M, N, pcb_module)
            return total_cycle_count, utilization

            
