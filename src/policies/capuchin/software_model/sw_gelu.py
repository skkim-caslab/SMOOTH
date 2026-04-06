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


@torch.compile
def gelu_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(input, approximate="tanh")


# x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return input

    def roofline_model(self, pcb_module: Device):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / min(pcb_module.io_module.bandwidth
            , pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq)
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):            
            self.M = M
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, ops_name: str):
        compute_cycle_count = 0
        io_cycle_count = 0
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(self.computational_graph.M / parallelism) * parallelism
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
           / pcb_module.compute_module.clock_freq
        )

        if 'collect' in ops_name:
            # Since both Load and Alloc tiles must exist in SRAM simultaneously,
            # the size of a single tile (tile_M * N * word_size) must be <= half (1/2) of the SRAM.
            # Since N=2 is passed in the function calls below, we set the denominator to 4 to adjust the size properly.
            max_elements_in_sram = pcb_module.compute_module.core.SRAM_size // (4 * data_type.word_size)
            tile_M = min(M, max_elements_in_sram)
            
            tile.collect_tile(
                tile_M, 2, -1, tile_M, 2, -1, pcb_module, ops_name, -1
            )
            tile.collect_alloc_tile(
                tile_M, 2, -1, tile_M, 2, -1, pcb_module, ops_name
            )

        else:
            sram_status, sram_table = sram.load_sram_status(pcb_module)
            is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name)
            write_or_free_ended = False
            unhided_io_amount = 0
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
                unhided_io_amount += loadable_amount - remained_amount
                is_loaded, needed_tile = sram.check_needed_tile_loaded(sram_status, 0, 0, 0, ops_name)


            compute_cycle_count = total_flop_count / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            io_cycle_count = io_latency * pcb_module.compute_module.clock_freq

            loadable_amount = compute_cycle_count * pcb_module.compute_module.l2_bandwidth_per_cycle / data_type.word_size 
            if(write_or_free_ended == False):
                remained_amount, sram_status, sram_stable = sram.write_previous_ops_from_sram(
                    sram_status, sram_table, ops_name, loadable_amount, pcb_module
                )
                write_or_free_ended = True
                loadable_amount = remained_amount

            while(loadable_amount != 0):
                loadable_amount, sram_status, sram_table, tot_find_overhead = sram.load_tile_to_sram(
                    sram_status, sram_table, pcb_module, loadable_amount
                )
            sram.store_sram_status(sram_status, sram_table)
            print("total cycle(X) : ", max(compute_cycle_count, total_io_count))
            print("compute cycle(X1) : ", compute_cycle_count)
            print("VE cycle(X1) : ", compute_cycle_count)
            print("io cycle(X2) : ", io_cycle_count)
            print("current cycle(X3) : ", max(compute_cycle_count, total_io_count))
            print("memory bw util[%](Y1) : ", 100)
            print("sram status: ", sram_status)
#            print("sram occupancy[%](Y2) : ", sram.get_sramutil(sram_status) / pcb_module.compute_module.core.SRAM_size *100)
            print("sa util[%](Y3) : ", 0)
            print("va util[%](Y3) : ", 100)


        return max(compute_cycle_count, io_cycle_count)

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # warmup
        for _ in range(3):
            _ = gelu_gpu(input)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = gelu_gpu(input)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        # print(latencies)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        import torch

        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = gelu_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
