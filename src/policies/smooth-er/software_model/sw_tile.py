from hardware_model.hw_device import Device
from math import ceil, log2, log
import json
import os
import multiprocessing
process_id = multiprocessing.current_process().pid

def collect_tile(
    M: int,
    N: int,
    K: int,
    l1_tile_M: int,
    l1_tile_N: int,
    l1_tile_K: int,
    pcb_module: Device,
    ops_name : str,
    dec_param : int, # For decision makeing in matrix multiplication, 2, 1, 0 means M_N, N_K, M_K
    data_size: int = -1, # For decision makeing in matrix multiplication, 2, 1, 0 means M_N, N_K, M_K
    ):
    file_path = f"./Tiles/whole_tile_list_{process_id}.json"
    ops_name = ops_name[:-8]

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if data_size != -1:
        size_to_append = data_size
    else:
        if dec_param == 2 or dec_param == -1:
            size_to_append = l1_tile_M * l1_tile_N
        elif dec_param == 1:
            size_to_append = l1_tile_N * l1_tile_K
        elif dec_param == 0:
            size_to_append = l1_tile_M * l1_tile_K
        else:
            size_to_append = l1_tile_M * l1_tile_N 


    size_to_append = int(size_to_append)
    if l1_tile_K != -1: # Matmul or softmax
        if dec_param == 2:
            data.append([ops_name + '_load_M_N_' + str(M) + '_' + str(N) + '_' + str(K) + '_', size_to_append])
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)


        elif dec_param == 1:
            data.append([ops_name + '_load_N_K_' + str(M) + '_' + str(N) + '_' + str(K) + '_', size_to_append])
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif dec_param == 0:
            data.append([ops_name + '_load_M_K_' + str(M) + '_' + str(N) + '_' + str(K) + '_', size_to_append])
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif dec_param == -1: # Softmax
            data.append([ops_name + '_load_M_N_' + str(M) + '_' + str(N) + '_' + str(K) + '_', size_to_append])
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

    else:
        l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
        iter_num = ceil(l1_tile_count / pcb_module.compute_module.core_count)

        for i in range(iter_num):
            data.append([ops_name + '_load', size_to_append])

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    return 0


def collect_alloc_tile(
    M: int,
    N: int,
    K: int,
    l1_tile_M: int,
    l1_tile_N: int,
    l1_tile_K: int,
    pcb_module: Device,
    ops_name : str,
    ):
    file_path = f"./Tiles/whole_tile_list_{process_id}.json"
    ops_name = ops_name[:-8]

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []


    if l1_tile_K != -1: # Matmul
        data.append([ops_name+'_alloc_M_N_K_' + str(M) + '_' + str(N) + '_' + str(K) + '_', l1_tile_M * l1_tile_N])

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        

    else:
        l1_tile_count = ceil(M/ l1_tile_M) * ceil(N / l1_tile_N)
        iter_num = ceil(l1_tile_count / pcb_module.compute_module.core_count)

        for i in range(iter_num):
            data.append([ops_name + '_alloc', l1_tile_M * l1_tile_N])
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    return 0
