from software_model.sw_transformer import(
#    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)

from software_model.sw_utils import data_type_dict, Tensor
from hardware_model.hw_system import system_dict

if __name__ == "__main__":

    bs = 1
    s = 1024
    sr_data_type = "int8"

    output_token_length = 1024

    TPU_system = system_dict["NPU_"+sr_data_type] 
    model = TransformerBlockAutoRegressionTP(
        d_model = 2048,
        n_heads=16,
        device_count=1,
        data_type=data_type_dict[sr_data_type],
        use_flash_attention=False,
        system=TPU_system
    )
    _ = model(
        Tensor([bs, 1, 2048], data_type_dict[sr_data_type]), s + output_token_length
    )
    model.compile_and_simulate(TPU_system)

