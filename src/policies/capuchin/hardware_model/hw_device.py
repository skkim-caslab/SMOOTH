from hardware_model.hw_modules import IOModule, IO_module_dict, MemoryModule, memory_module_dict, ComputeModule, compute_module_dict

class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_module: MemoryModule,
    ) -> None:
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module


device_dict = {
    "NPU_int8_512KB_2048": Device(
        compute_module_dict["NPU_int8_512KB_2048"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_1024": Device(
        compute_module_dict["NPU_int8_8MB_1024"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_2048": Device(
        compute_module_dict["NPU_int8_8MB_2048"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_4096": Device(
        compute_module_dict["NPU_int8_8MB_4096"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_64": Device(
        compute_module_dict["NPU_int8_8MB_64"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_32MB_64": Device(
        compute_module_dict["NPU_int8_32MB_64"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),


    "NPU_int8_2MB_64": Device(
        compute_module_dict["NPU_int8_2MB_64"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_1280": Device(
        compute_module_dict["NPU_int8_8MB_1280"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_1536": Device(
        compute_module_dict["NPU_int8_8MB_1536"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_2560": Device(
        compute_module_dict["NPU_int8_8MB_2560"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_3072": Device(
        compute_module_dict["NPU_int8_8MB_3072"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_5120": Device(
        compute_module_dict["NPU_int8_8MB_5120"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_64MB_2048": Device(
        compute_module_dict["NPU_int8_64MB_2048"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_512KB_1": Device(
        compute_module_dict["NPU_int8_512KB_1"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_8MB_1": Device(
        compute_module_dict["NPU_int8_8MB_1"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_int8_64MB_1": Device(
        compute_module_dict["NPU_int8_64MB_1"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_fp16": Device(
        compute_module_dict["NPU_fp16"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_fp32": Device(
        compute_module_dict["NPU_fp32"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
}
