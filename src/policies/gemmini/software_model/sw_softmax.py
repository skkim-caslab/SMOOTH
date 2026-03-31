
from utils import size
from typing import List, Tuple
from hardware_model.hw_device import Device
from software_model.sw_operators import Operator
from software_model.sw_utils import Tensor, DataType
from math import ceil, log2
import torch
import time
import statistics
import numpy as np
import json
import os


class Softmax(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None
        self.tile_config = None

    def __call__(self, input: Tensor, seq_len: int, config_file) -> Tensor:
        assert self.data_type == input.data_type
        self.seq_len = seq_len
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
#        print(f'SKKIM softmax dimension: {self.M}, {self.N}')
        with open(config_file, 'r') as f:
            self.tile_config = json.load(f)

        return input

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
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, is_l2_double_buffering: {self.is_l2_double_buffering}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, is_l1_double_buffering: {self.is_l1_double_buffering}"
            )
    
    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 3
        self.flop_count = self.M * self.N * (pcb_module.compute_module.core.vector_unit.flops_per_exp * 3 + 7)
        self.roofline_latency=max(self.io_count/min(pcb_module.io_module.bandwidth, pcb_module.compute_module.l2_bandwidth_per_cycle*pcb_module.compute_module.clock_freq), self.flop_count/pcb_module.compute_module.total_vector_flops)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = M
#        l2_tile_M = (
#            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
#        )
#        l2_tile_M = min(l2_tile_M, M)

#        print('l2_tile_M, l2_tile_N : ', l2_tile_M, l2_tile_N)
        is_l2_double_buffering = False
#        for l1_N_tiling_factor in [16]:
#            for l1_tile_M in [8]:
        l1_tile_M = self.tile_config['softmax']['l1_tile_M'] #16
        l1_tile_N = self.tile_config['softmax']['l1_tile_N'] #1024

        cache_config = f"configs/softmax_{pcb_module.compute_module.l2_bandwidth_per_cycle}_{pcb_module.compute_module.core.SRAM_size}_{self.M}_{self.N}_{l1_tile_M}_{l1_tile_N}.cfg"
        '''
        if os.path.exists(cache_config):
            with open(cache_config, 'r') as f:
                cached_data = json.load(f)
                self.best_cycle_count = cached_data.get('best_cycle_count', 0)
                self.latency = cached_data.get('latency', self.latency)
                print("total cycle:",self.best_cycle_count)
                return self.best_cycle_count
                #return self.latency
        '''



        #l1_tile_N = 3073
        is_l1_double_buffering = True
        if is_l1_double_buffering:
            assert (
                l1_tile_M * l1_tile_N * data_type.word_size
                <= pcb_module.compute_module.core.SRAM_size // 2
            ), "L1 double buffering tile size exceeds half of the SRAM size"
        else:
            assert (
                l1_tile_M * l1_tile_N * data_type.word_size
                <= pcb_module.compute_module.core.SRAM_size
            ), "L1 tile size exceeds the SRAM size"
        mapping = self.Mapping(
            l2_tile_M,
            l2_tile_N,
            is_l2_double_buffering,
            l1_tile_M,
            l1_tile_N,
            is_l1_double_buffering,
        )
        cycle_count = self.simulate(
            self.computational_graph, mapping, pcb_module
        )
        if cycle_count < min_cycle_count:
            min_cycle_count = cycle_count
            best_mapping = mapping
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        self.best_mapping.display()
        M_size = self.best_mapping.l1_tile_M
        N_size = self.best_mapping.l1_tile_N

        cached_data = {
            'best_cycle_count': self.best_cycle_count,
            'latency': self.latency,
        }
        with open(cache_config, 'w') as f:
            json.dump(cached_data, f, indent=4)

        return self.best_cycle_count
        #return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
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

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
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
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
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
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            # Cycle count variables
            total_read_byte = 0
            total_read_cycle_count = 0
            total_compute_cycle_count = 0
            total_write_byte = 0
            total_write_cycle_count = 0
            skkim_total_cycle_count = 0



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
            time_tick = 0
            skkim_check_dict={}
            #for i in range(ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1):
            for i in range(ceil(l1_tile_count / pcb_module.compute_module.core_count)):
#                print("SKKIM cnt", i)
                time_tick += l1_tile_cycle_count
                skkim_total_cycle_count = total_cycle_count
                #skkim: can optimize with hiding skkim_read_cycle_count
                print("total cycle(X) : ",  time_tick)
                print("compute cycle(X1) : ", l1_tile.compute_cycle_count)
                print("VE cycle(X1) : ", l1_tile.compute_cycle_count)
                print("io cycle(X2) : ", l1_tile.read_cycle_count + l1_tile.write_cycle_count)
#                print("io cycle; read(X2) : ", l1_tile.read_cycle_count)
#                print("io cycle; write(X2) : ", l1_tile.write_cycle_count)
                print("current cycle(X3) : ",  l1_tile_cycle_count)
                input_check = (l1_tile.compute_cycle_count, l1_tile.read_cycle_count + l1_tile.write_cycle_count,l1_tile_cycle_count)
                if input_check in skkim_check_dict:
                    skkim_check_dict[input_check] += 1
                else: 
                    skkim_check_dict[input_check] = 1
                loadable_amount =  l1_tile_cycle_count - (l1_tile.read_cycle_count + l1_tile.write_cycle_count)
                if loadable_amount != 0:
                    print("loadable cycle(X4) : ", loadable_amount)
                print("sram occupancy[%](Y2) : ",  (l1_tile_M * l1_tile_N * data_type.word_size * 2)/pcb_module.compute_module.core.SRAM_size * 100)
                print("memory bw util[%](Y1) : ", (l1_tile.read_cycle_count + l1_tile.write_cycle_count)/l1_tile_cycle_count*100 )
                print("sa util[%](Y3) : ", 0 )
                print("va util[%](Y3) : ", l1_tile.va_util * 100 )

#            print("SKKIM", skkim_check_dict)
            return time_tick
            #return total_cycle_count


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
            # skkim: want to operate softamx parallel. then need to consider vector width and vector count
#            M_per_vector_count = ceil(
#                M / pcb_module.compute_module.core.vector_unit.vector_count
#            )
#            N_per_vector_count = N
#            M_per_vector_lane = M_per_vector_count
#            N_per_vector_lane = ceil(
#                N_per_vector_count
#                / pcb_module.compute_module.core.vector_unit.vector_width
#            )
#            total_flop_count = M_per_vector_lane * N_per_vector_lane * (self.flops_per_exp * 3 + 7)
            total_cycle_count = ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )
            utilization = self.calculate_softmax_vector_utilization(total_cycle_count, M, N, pcb_module)

#            print("SKKIM softmax", M,N, total_cycle_count)
            return utilization, total_cycle_count

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
