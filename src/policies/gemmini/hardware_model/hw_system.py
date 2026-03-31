from hardware_model.hw_device import Device, device_dict
from hardware_model.hw_interconnect import InterConnectModule, interconnect_module_dict

class System:
    def __init__(self, pcb_module: Device, interconnect: InterConnectModule) -> None:
        self.device = pcb_module
        self.interconnect = interconnect


system_dict = {
    "A100_4_fp16": System(
        device_dict["A100_80GB_fp16"],
        interconnect_module_dict["NVLinkV3_FC_4"],
    ),
    "NPU_fp32": System(device_dict["NPU_fp32"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_fp16": System(device_dict["NPU_fp16"], interconnect_module_dict["TPUv3Link_8"]),   
    "NPU_int8": System(device_dict["NPU_int8"], interconnect_module_dict["TPUv3Link_8"])    
}
