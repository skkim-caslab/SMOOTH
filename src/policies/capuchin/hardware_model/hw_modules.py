from math import ceil
from software_model.sw_utils import DataType, data_type_dict

#### Compute_Modules ####

class VectorUnit:
    def __init__(
        self,
        total_vector_flops_per_cycle,
        word_size,
        flops_per_exp,
        vector_width,
        vector_count,
        data_type=data_type_dict["int8"],
    ):
        self.total_vector_flops_per_cycle = total_vector_flops_per_cycle
        self.word_size = word_size  # Byte
        self.flops_per_exp = flops_per_exp  # flops per exp instruction
        self.vector_width = vector_width
        self.vector_count = vector_count
        self.flops_per_cycle = ceil(
            total_vector_flops_per_cycle / vector_width / vector_count
        )
        self.data_type = data_type


vector_unit_dict = {
    "NPU_int8": VectorUnit(512, 1, 15, 32, 32, data_type_dict["int8"]),
    "NPU_fp16": VectorUnit(512, 2, 15, 32, 32, data_type_dict["fp16"]),
    "NPU_fp32": VectorUnit(512, 4, 15, 32, 32, data_type_dict["fp32"]),
}


class SystolicArray:
    def __init__(
        self,
        array_height,
        array_width,
        mac_per_cycle,
        input_word_size,
        output_word_size,
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size


systolic_array_dict = {
    "NPU_fp32": SystolicArray(32, 32, 1, 4, 4),
    "NPU_fp16": SystolicArray(32, 32, 2, 2, 2),
    "NPU_int8": SystolicArray(32, 32, 4, 1, 1),
}


class Core:
    def __init__(
        self,
        vector_unit: VectorUnit,
        systolic_array: SystolicArray,
        systolic_array_count,
        SRAM_size,
        block_size,
    ):
        self.vector_unit = vector_unit
        self.systolic_array = systolic_array
        self.systolic_array_count = systolic_array_count
        self.SRAM_size = SRAM_size  # Byte
        self.block_size = block_size  # Byte
        # assert(vector_unit.word_size==systolic_array.word_size)
        self.vector_word_size = vector_unit.word_size


core_dict = {
    "Core_NPU_fp32": Core(
        vector_unit_dict["NPU_fp32"],
        systolic_array_dict["NPU_fp32"],
        1,
        512 * 1024,
        2048, #block size
    ),
    "Core_NPU_fp16": Core(
        vector_unit_dict["NPU_fp16"],
        systolic_array_dict["NPU_fp16"],
        1,
        512 * 1024,
        2048, #block size
    ),
    "Core_NPU_int8_512KB_2048": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        512 * 1024, #SRAM size
        2048, #block size
    ),
    "Core_NPU_int8_8MB_4096": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024, #SRAM size
        4096, #block size
    ),
    "Core_NPU_int8_2MB_64": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        2 * 1024 * 1024,  # SRAM size
        64,             # block size
    ),
    "Core_NPU_int8_32MB_64": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        32 * 1024 * 1024,  # SRAM size
        64,             # block size
    ),

    "Core_NPU_int8_8MB_64": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        64,             # block size
    ),
    "Core_NPU_int8_8MB_1280": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        1280,             # block size
    ),
    "Core_NPU_int8_8MB_1536": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        1536,             # block size
    ),
    "Core_NPU_int8_8MB_2560": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        2560,             # block size
    ),
    "Core_NPU_int8_8MB_3072": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        3072,             # block size
    ),
    "Core_NPU_int8_8MB_5120": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024,  # SRAM size
        5120,             # block size
    ),
    "Core_NPU_int8_8MB_2048": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024, #SRAM size
        2048, #block size
    ),
    "Core_NPU_int8_8MB_1024": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024, #SRAM size
        1024, #block size
    ),
    "Core_NPU_int8_64MB_2048": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        128 * 512 * 1024, #SRAM size
        2048, #block size
    ),
    "Core_NPU_int8_512KB_1": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        512 * 1024, #SRAM size
        1, #block size
    ),
    "Core_NPU_int8_8MB_1": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        16 * 512 * 1024, #SRAM size
        1, #block size
    ),
    "Core_NPU_int8_64MB_1": Core(
        vector_unit_dict["NPU_int8"],
        systolic_array_dict["NPU_int8"],
        1,
        128 * 512 * 1024, #SRAM size
        1, #block size
    ),
}
# compute_tile_dict={'SM_A100_int8':ComputeTile(512, 4096, 192*1024*8,3.41, 'TSMC N7', 128*8),'SM_A100_fp16':ComputeTile(512, 2048, 192*1024*8,3.41, 'TSMC N7', 128),}
# flops: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch__fig2
# area: https://pbs.twimg.com/media/FOT_-NJWUAARrtB?format=jpg&name=large

class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu, flash_attention):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu
        self.flash_attention = flash_attention


overhead_dict = {
    "NPU": Overhead(0,0,0,0,0),
}


class ComputeModule:
    def __init__(
        self,
        core: Core,
        core_count,
        clock_freq,
        l2_size,
        l2_bandwidth_per_cycle,
        overhead: Overhead = overhead_dict["NPU"],
    ):
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)  # Byte
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle  # Byte/clock
        self.total_vector_flops_per_cycle = (
            core.vector_unit.total_vector_flops_per_cycle * core_count
        )
        self.total_vector_flops = self.total_vector_flops_per_cycle * clock_freq
        self.total_systolic_array_flops = (
            core_count
            * core.systolic_array_count
            * core.systolic_array.mac_per_cycle
            * 2
            * core.systolic_array.array_height
            * core.systolic_array.array_width
            * clock_freq
        )
        self.overhead = overhead


compute_module_dict = {
    "NPU_int8_512KB_1": ComputeModule(
        core_dict["Core_NPU_int8_512KB_1"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_1": ComputeModule(
        core_dict["Core_NPU_int8_8MB_1"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_64MB_1": ComputeModule(
        core_dict["Core_NPU_int8_64MB_1"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_512KB_2048": ComputeModule(
        core_dict["Core_NPU_int8_512KB_2048"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_4096": ComputeModule(
        core_dict["Core_NPU_int8_8MB_4096"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_32MB_64": ComputeModule(
        core_dict["Core_NPU_int8_32MB_64"],
        1,
        940e6,
        12 * 1024**3,  # LPDDR5
        32,            # MemBW (GB/s)
        overhead_dict["NPU"],
    ),

    "NPU_int8_8MB_64": ComputeModule(
        core_dict["Core_NPU_int8_8MB_64"],
        1,
        940e6,
        12 * 1024**3,  # LPDDR5
        32,            # MemBW (GB/s)
        overhead_dict["NPU"],
    ),
    "NPU_int8_2MB_64": ComputeModule(
        core_dict["Core_NPU_int8_2MB_64"],
        1,
        940e6,
        12 * 1024**3,  # LPDDR5
        32,            # MemBW (GB/s)
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_1280": ComputeModule(
        core_dict["Core_NPU_int8_8MB_1280"],
        1,
        940e6,
        12 * 1024**3,  # LPDDR5
        32,            # MemBW (GB/s)
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_1536": ComputeModule(
        core_dict["Core_NPU_int8_8MB_1536"],
        1,
        940e6,
        12 * 1024**3,
        32,
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_2560": ComputeModule(
        core_dict["Core_NPU_int8_8MB_2560"],
        1,
        940e6,
        12 * 1024**3,
        32,
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_3072": ComputeModule(
        core_dict["Core_NPU_int8_8MB_3072"],
        1,
        940e6,
        12 * 1024**3,
        32,
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_5120": ComputeModule(
        core_dict["Core_NPU_int8_8MB_5120"],
        1,
        940e6,
        12 * 1024**3,
        32,
        overhead_dict["NPU"],
    ),

    "NPU_int8_8MB_2048": ComputeModule(
        core_dict["Core_NPU_int8_8MB_2048"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_8MB_1024": ComputeModule(
        core_dict["Core_NPU_int8_8MB_1024"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_int8_64MB_2048": ComputeModule(
        core_dict["Core_NPU_int8_64MB_2048"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_fp16": ComputeModule(
        core_dict["Core_NPU_fp16"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
    "NPU_fp32": ComputeModule(
        core_dict["Core_NPU_fp32"],
        1,
        940e6,
        12 * 1024**3, ##LPDDR5
        32, ##MemBW
        overhead_dict["NPU"],
    ),
}



#### Memory Module ####
class MemoryModule:
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity

memory_module_dict = {'NPU': MemoryModule(float('inf'))}


#### IO Module ####
class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency


IO_module_dict = {
    "NPU": IOModule(float("inf"), 1e-6),
}
