from software_model.sw_transformer import(
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)

from software_model.sw_utils import data_type_dict, Tensor
from hardware_model.hw_system import system_dict
import sys

if __name__ == "__main__":

    bs = 1
    INITFLAG = False
    
    config_file = sys.argv[2]
    model_dim = int(sys.argv[3])
    head_num = int(sys.argv[4])
    INITFLAG = "--init" in sys.argv

    
    sr_data_type = "int8"

    print("Generation")
    if len(sys.argv) < 3:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    TPU_system = system_dict["NPU_"+sr_data_type] 

    if INITFLAG == True:
        input_token_length = int(sys. argv[1])
        model = TransformerBlockInitComputationTP(
            d_model = model_dim,
            n_heads=head_num,
            device_count=1,
            data_type=data_type_dict[sr_data_type],
            system=TPU_system,
            config_file = config_file,
            use_flash_attention=False,
        )
        _ = model(
            Tensor([bs, input_token_length, model_dim], data_type_dict[sr_data_type])
        )
        model.compile_and_simulate(TPU_system)
    else:
        input_token_length = 1024
        output_token_length = int(sys. argv[1])
        model = TransformerBlockAutoRegressionTP(
            d_model = model_dim,
            n_heads=head_num,
            device_count=1,
            data_type=data_type_dict[sr_data_type],
            system=TPU_system,
            config_file = config_file,
            use_flash_attention=True,
        )
        _ = model(
            Tensor([bs, 1, model_dim], data_type_dict[sr_data_type]), input_token_length + output_token_length
        )
        model.compile_and_simulate(TPU_system)
