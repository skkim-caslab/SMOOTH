from hardware_model.hw_device import Device, device_dict
from hardware_model.hw_interconnect import InterConnectModule, interconnect_module_dict

class System:
    def __init__(self, pcb_module: Device, interconnect: InterConnectModule) -> None:
        self.device = pcb_module
        self.interconnect = interconnect


system_dict = {
    "NPU_fp32": System(device_dict["NPU_fp32"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_fp16": System(device_dict["NPU_fp16"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_512KB_2048": System(device_dict["NPU_int8_512KB_2048"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_4096": System(device_dict["NPU_int8_8MB_4096"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_8MB_2048": System(device_dict["NPU_int8_8MB_2048"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_8MB_1024": System(device_dict["NPU_int8_8MB_1024"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_8MB_1280": System(device_dict["NPU_int8_8MB_1280"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_64": System(device_dict["NPU_int8_8MB_64"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_32MB_64": System(device_dict["NPU_int8_32MB_64"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_2MB_64": System(device_dict["NPU_int8_2MB_64"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_2MB_1024": System(device_dict["NPU_int8_2MB_1024"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_32MB_1024": System(device_dict["NPU_int8_32MB_1024"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_1536": System(device_dict["NPU_int8_8MB_1536"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_2560": System(device_dict["NPU_int8_8MB_2560"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_3072": System(device_dict["NPU_int8_8MB_3072"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_5120": System(device_dict["NPU_int8_8MB_5120"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_64MB_2048": System(device_dict["NPU_int8_64MB_2048"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_512KB_1": System(device_dict["NPU_int8_512KB_1"], interconnect_module_dict["TPUv3Link_8"]),
    "NPU_int8_8MB_1": System(device_dict["NPU_int8_8MB_1"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8_64MB_1": System(device_dict["NPU_int8_64MB_1"], interconnect_module_dict["TPUv3Link_8"]),   
}
