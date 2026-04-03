from utils import size
from typing import List, Tuple
from hardware_model.hw_device import Device
from software_model.sw_operators import Operator
from software_model.sw_utils import Tensor, DataType
import software_model.sw_sramcontroller as sram
import software_model.sw_tile as tile
from math import ceil, log2
import torch
import time
import statistics
import numpy as np
import json


class Softmax(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None
        self.tile_config = None

    def __call__(self, input: Tensor, seq_len:int, config_file) -> Tensor:
        assert self.data_type == input.data_type
        self.seq_len = seq_len
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        with open(config_file, 'r') as f:
            self.tile_config = json.load(f)
        return input

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int):
        for m in range(loop_M):
            for n in range(loop_N):
                yield m, n

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
    
    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 3
        self.flop_count = self.M * self.N * (pcb_module.compute_module.core.vector_unit.flops_per_exp * 3 + 7)
        self.roofline_latency=max(self.io_count/min(pcb_module.io_module.bandwidth, pcb_module.compute_module.l2_bandwidth_per_cycle*pcb_module.compute_module.clock_freq), self.flop_count/pcb_module.compute_module.total_vector_flops)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, ops_name):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
        )
        l2_tile_M = min(l2_tile_M, M)

        is_l2_double_buffering = False
        l2_tile_M = M
        l2_tile_N = N
        is_l2_double_buffering = False
        l1_tile_M = self.tile_config['softmax']['l1_tile_M'] #1
        l1_tile_N = self.tile_config['softmax']['l1_tile_N'] #3073 
        is_l1_double_buffering = True

        mapping = self.Mapping(
            l2_tile_M,
            l2_tile_N,
            is_l2_double_buffering,
            l1_tile_M,
            l1_tile_N,
            is_l1_double_buffering,
        )
        cycle_count = self.simulate(
            self.computational_graph, mapping, pcb_module, ops_name
        )
        min_cycle_count = cycle_count
        best_mapping = mapping

        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count 
        self.latency = self.best_latency
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N
        is_l2_double_buffering = self.best_mapping.is_l2_double_buffering
        l2_tile_M = self.best_mapping.l2_tile_M
        l2_tile_N = self.best_mapping.l2_tile_N
        is_l1_double_buffering = self.best_mapping.is_l1_double_buffering

        
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

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
                ops_name,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
                ops_name,
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
            mapping: "Softmax.Mapping",
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
            mapping: "Softmax.Mapping",
            pcb_module: Device,
            ops_name: str,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = Softmax.L1TileSimulator(
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
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
                l1_tile_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )

            for m, n in Softmax.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
            ):
                tile.collect_tile(
                    m, n, 0, l1_tile_M, l1_tile_N, 0, pcb_module, ops_name, -1
                )

            return total_cycle_count


        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
            ops_name: str,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = Softmax.L1TileSimulator(
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


            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
                l1_tile_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )

            sram_status, sram_table = sram.load_sram_status(pcb_module)
            previous_m_n_k =  '_0_0_0'
            time_tick = 0
            for m, n in Softmax.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
            ):
                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, m, n, 0, ops_name)
                write_or_free_ended = False
                while(is_loaded == False): 
                    loadable_amount = pcb_module.compute_module.core.SRAM_size
                    if(write_or_free_ended):
                        remained_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                            sram_status, sram_table, pcb_module, loadable_amount
                        )
                    else:
                        if (m, n) == (0, 0):
                            remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                                sram_status, sram_table, ops_name, loadable_amount, pcb_module
                            )
                        else:
                            remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, pcb_module, ops_name, 2, loadable_amount
                            )
                        write_or_free_ended = True

                    is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, m, n, 0, ops_name)
                    time.sleep(2)
                vector_repeat_time = 1
                remained_amount_pre = 0
                base_cycle_count = l1_tile.compute_cycle_count // vector_repeat_time
                remainder_cycle_count = l1_tile.compute_cycle_count % vector_repeat_time


                for vector_unit_compute in range(vector_repeat_time):
                    if vector_unit_compute == vector_repeat_time - 1:
                        current_compute_cycle = base_cycle_count + remainder_cycle_count
                    else:
                        current_compute_cycle = base_cycle_count


                    current_time =  current_compute_cycle +(l1_tile.read_cycle_count + l1_tile.write_cycle_count)/vector_repeat_time
                    time_tick += current_time
                    print('total cycle(X) : ', time_tick)
                    print('compute cycle(X1) : ', current_compute_cycle)
                    print('VE cycle(X1) : ', current_compute_cycle)
                    loadable_amount = current_compute_cycle * pcb_module.compute_module.l2_bandwidth_per_cycle / pcb_module.compute_module.core.systolic_array.input_word_size + remained_amount_pre
                    if (m, n) == (0, 0):
                        remained_amount, sram_status, sram_table = sram.write_previous_ops_from_sram(
                            sram_status, sram_table, ops_name, loadable_amount, pcb_module
                        )
                        if vector_repeat_time == 1:
                            loadable_amount = remained_amount
                        else:
                            loadable_amount = remained_amount
                            sram_status, sram_table = sram.free_ended_block_from_sram(
                            sram_status, sram_table, previous_m_n_k, pcb_module, ops_name, 2
                        )
                    else:
                        if vector_repeat_time == 1:
                            remained_amount, sram_status, sram_table = sram.write_tile_from_sram(
                                sram_status, sram_table, previous_m_n_k, pcb_module, ops_name, 2, loadable_amount
                            )
                        else:
                            sram_status, sram_table = sram.free_ended_block_from_sram(
                                sram_status, sram_table, previous_m_n_k, pcb_module, ops_name, 2
                            )
                    loadable_amount = remained_amount
                    if loadable_amount == float("inf"):
                        print_io_cycle = current_time
                    else:
                        print_io_cycle =  ceil(loadable_amount * pcb_module.compute_module.core.systolic_array.input_word_size / pcb_module.compute_module.l2_bandwidth_per_cycle)
  
                    print('io cycle(X2) : ', print_io_cycle)
                    print('current cycle(X3) : ', current_time)
                    print('memory bw util[%](Y1) : 100' )
                    print('va util[%](Y3) : 100' )
                    print('sa util[%](Y3) : 0' )

                    while(loadable_amount != 0):
                        loadable_amount, sram_status, sram_table, remained_amount_pre = sram.load_tile_to_sram(
                            sram_status, sram_table, pcb_module, loadable_amount
                        )

                    print('sram occupancy[%] :', sram.get_sramutil(sram_status, pcb_module)/pcb_module.compute_module.core.SRAM_size * 100)
                    previous_m_n_k = '_' + str(m) + '_' + str(n) + '_0'

            # If the N / l1_tile_N value changes, you may need to pay attention to the reduction cycle count.

            sram.store_sram_status(sram_status, sram_table)

            return time_tick


    class L1TileSimulator:
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

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.softmax(input, dim=-1)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.softmax(input, dim=-1)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.softmax(a, dim=-1)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
