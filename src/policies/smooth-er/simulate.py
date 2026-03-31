from software_model.sw_transformer import(
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)

from software_model.sw_utils import data_type_dict, Tensor
from hardware_model.hw_system import system_dict
import sys
import json

if __name__ == "__main__":

    bs = 1
    INITFLAG = False

    #config_file_path = 'configs/generated_configs/1B_w4a8_base_8MB_b1K.json'
    sram_size = "8MB"
    #model_dim = 2048
    #head_num = 32

    #with open(config_file_path, 'r') as f:
    #    execution_cfg = json.load(f)

#    input_token_length = int(sys. argv[1])
#    output_token_length = 8192
    config_file = sys.argv[2]
    model_dim = int(sys.argv[3])
    head_num = int(sys.argv[4])
    INITFLAG = "--init" in sys.argv


    #quant = execution_cfg.get("quant", "w4a8")  # 기본값: w4a8
    #sram_size = execution_cfg.get("sram_size", "8MB")
    #block_size = execution_cfg.get("block_size", "4096")
    #block_size = model_dim
    block_size = 1024



    sr_data_type = "int8"

#    if quant == "w4a8":
#        model_dim = model_dim//2

#    config_file = './Tiles/tile_size.json'
    if len(sys.argv) < 3:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[2]


    print("Generation")
#    s = 1024

    TPU_system = system_dict["NPU_"+sr_data_type+"_"+sram_size+"_"+str(block_size)] 

    if INITFLAG == True:
        input_token_length = int(sys. argv[1])
        model = TransformerBlockInitComputationTP(
            d_model = model_dim,
            n_heads=head_num,  #1.3B 16, 6.7B 32
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
        output_token_length = int(sys. argv[1])
        input_token_length = 1024
        model = TransformerBlockAutoRegressionTP(
            d_model = model_dim,
            n_heads=head_num,
            device_count=1,
            data_type=data_type_dict[sr_data_type],
            system=TPU_system,
            config_file = config_file,
            use_flash_attention=True,
            #use_flash_attention=execution_cfg.get("flash_attention", False),
        )
        _ = model(
            Tensor([bs, 1, model_dim], data_type_dict[sr_data_type]), input_token_length + output_token_length
        )
        model.compile_and_simulate(TPU_system)

