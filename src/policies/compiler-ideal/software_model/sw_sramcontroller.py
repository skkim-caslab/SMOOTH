from hardware_model.hw_device import Device
from math import ceil, log2, log
import json
import os
import fcntl
import multiprocessing
process_id = multiprocessing.current_process().pid
current_reg=0

def load_sram_status():
    sram_status = []
    process_dir = f"./SRAM"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"sram_status_{process_id}.json")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                sram_status = json.load(f)
            except json.JSONDecodeError:
                sram_status = []
    else:
        sram_status = []
    #print(sram_status)
    return sram_status
def load_tile_to_sram(
    sram_status: list,
    pcb_module: Device,
    loadable_amount : int,
    ):

    process_dir = f"./Tiles/"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
#    file_path = "./Tiles/remained_tile_list.json"
    try:
        with open(file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
    except (json.JSONDecodeError, FileNotFoundError):
        data = []
    #with open(file_path, 'r') as f:
    #    try:
    #        data = json.load(f)
    #    except json.JSONDecodeError:
    #        data = []
    #        print("Json Error")

    num_of_tile = len(sram_status)

    if len(data) == 0:
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
        return 0, sram_status


    if num_of_tile == 0: #SRAM에 어떤 타일도 없는 경우
        start_addr = 0
        if 'alloc' in data[0][0]:
            if data[0][1] > pcb_module.compute_module.core.SRAM_size:
                raise Exception("Alloc tile size exceed SRAM size")

            else:
                end_addr = data[0][1] - 1
                sram_status.append([data[0][0], start_addr, end_addr, 1])
                #print("SKKIM 1.", start_addr, end_addr)
                sram_status = sort_sram_status(sram_status)
                data.pop(0)

        else:
            if loadable_amount >= data[0][1]:
                if data[0][1] > pcb_module.compute_module.core.SRAM_size:
                    raise Exception("Load tile size exceed SRAM size.")

                else:
                    end_addr = data[0][1] - 1
                    sram_status.append([data[0][0], start_addr, end_addr, 1])
                    #print("SKKIM 2.", start_addr, end_addr)
                    sram_status = sort_sram_status(sram_status)
                    loadable_amount = loadable_amount - data[0][1]
                    data.pop(0)
                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                            json.dump(data, f, indent=2)
                            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                        return 0, sram_status

                    if 'alloc' in data[0][0]:
                        start_addr = end_addr + 1
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else:
                            sram_status.append([data[0][0], start_addr, end_addr , 1])
                            #print("SKKIM 3.", start_addr, end_addr)
                            sram_status = sort_sram_status(sram_status)
                            data.pop(0)


        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제


    else:
        start_addr = sram_status[num_of_tile-1][2] + 1
        for i in range(len(sram_status)):
            if sram_status[i][3] == 0: #SRAM에 transfer 중 끊긴 타일이 있다.(이 타일은 alloc tile은 아니다.)
                if loadable_amount >= data[0][1]:
                    sram_status[i][3] = 1
                    sram_status[i][2] = sram_status[i][2] + data[0][1]
                    loadable_amount = loadable_amount - data[0][1]
                    data.pop(0)
                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                            json.dump(data, f, indent=2)
                            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                        return 0, sram_status

                    if 'alloc' in data[0][0]:
                        start_addr = sram_status[i][2] + 1
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else:
                            sram_status.append([data[0][0], start_addr, end_addr, 1])
                            #print("SKKIM 4.", start_addr, end_addr)
                            sram_status = sort_sram_status(sram_status)
                            data.pop(0)
                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                                    json.dump(data, f, indent=2)
                                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                                return 0, sram_status
                else:
                    org_loaded = sram_status[i][2]
                    sram_status[i][2] = sram_status[i][2] + loadable_amount
                    #print("SKKIM 5.", org_loaded+1, sram_status[i][2])
                    data[0][1] = data[0][1] - loadable_amount
                    loadable_amount = 0

                with open(file_path, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제

                break

            if i == len(sram_status)-1: #SRAM에 transfer가 중단된 타일이 없는 경우
                if 'alloc' in data[0][0]: #만약 다음 load해야할 타일이 alloc 타일인 경우
                    #print("SKKIM for loop 0")
                    start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                    if (start_addr, end_addr) == (-1, -1):
                        loadable_amount = 0
###
                        for i in range(len(sram_status)):
                            if 'alloc' in sram_status[i][0]: 
                                break
                            if i == len(sram_status) - 1:
                                #print("Free All loaded Tile")
                                #print("SRAM Status", sram_status)
                                process_dir = f"./Tiles/"
                                os.makedirs(process_dir, exist_ok=True)

                                file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
                                #file_path = "./Tiles/remained_tile_list.json"
                                with open(file_path, 'r') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                                    data = json.load(f)
                                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                                for item in sram_status:
                                    data.insert(0, [item[0], item[2] - item[1]])
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                                    json.dump(data, f, indent = 2)
                                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                                sram_status = []

                    else:
                        sram_status.append([data[0][0], start_addr, end_addr, 1])
                        #print("SKKIM 6.", start_addr, end_addr)
                        sram_status = sort_sram_status(sram_status)
                        data.pop(0)


                else:
                    #print("SKKIM for loop 1")
                    if loadable_amount >= data[0][1]: #loadable_amount가 다음 tile을 충분히 다 load할 수 있다면
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else: #SRAM에 해당 타일을 다 넣을 수 있는 자리가 있다면,
                            sram_status.append([data[0][0], start_addr, end_addr, 1])
                            #print("SKKIM 7.", start_addr, end_addr)
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = loadable_amount - data[0][1]
                            data.pop(0)
                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                                    json.dump(data, f, indent=2)
                                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
                                return 0, sram_status

                            if 'alloc' in data[0][0]:
                                start_addr = end_addr + 1
                                start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                                if (start_addr, end_addr) == (-1, -1):
                                    loadable_amount = 0

                                else:
                                    sram_status.append([data[0][0], start_addr, end_addr , 1])
                                    #print("SKKIM 8.", start_addr, end_addr)
                                    sram_status = sort_sram_status(sram_status)
                                    data.pop(0)
                    else: #loadable_amount가 다음 타일을 충분히 다 load할 수 없다면
                        loadable_amount = 0
                        #start_addr, end_addr = addr_decider(sram_status, start_addr, loadable_amount, data[0][1], pcb_module)
                        #loadable_amount = 0
                        #if (start_addr, end_addr) == (-1, -1):
                        #    loadable_amount = 0

                        #else:
                        #    sram_status.append([data[0][0], start_addr, end_addr, 0])
                            #print("SKKIM 9.", start_addr, end_addr)
                        #    sram_status = sort_sram_status(sram_status)
                        #    data[0][1] = data[0][1] - loadable_amount
                        #    loadable_amount = 0


                with open(file_path, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제


    return loadable_amount, sram_status

'''
def load_tile_to_sram( #skkim: load multiple tile
    sram_status: list,
    pcb_module: Device,
    loadable_amount: int,
):
    process_dir = f"./Tiles/"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
    try:
        with open(file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
    except (json.JSONDecodeError, FileNotFoundError):
        data = []

    num_of_tile = len(sram_status)

    if len(data) == 0:
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제
        return 0, sram_status

    # Group tiles by base operation and M,N,K
    def group_tiles_by_operation(tiles):
        grouped = {}
        for tile in tiles:
            tile_name = tile[0]
            parts = tile_name.split('_')
            if len(parts) > 3:
                # Extract base operation name (before 'load' or 'alloc')
                if 'load' in tile_name or 'alloc' in tile_name:
                    for i, part in enumerate(parts):
                        if part in ['load', 'alloc']:
                            op_name = '_'.join(parts[:i])
                            break
                    m, n, k = parts[-4:-1]
                    key = f"{op_name}_{m}_{n}_{k}"
                else:
                    key = tile_name
            else:
                key = tile_name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(tile)
        return grouped

    grouped_tiles = group_tiles_by_operation(data)
    if not grouped_tiles:
        return 0, sram_status

    # Process the first group of tiles
    first_key = next(iter(grouped_tiles))
    tiles_to_load = grouped_tiles[first_key]
    total_size = sum(tile[1] for tile in tiles_to_load)

    # Check if all tiles in the group can be loaded
    if total_size > loadable_amount or total_size > pcb_module.compute_module.core.SRAM_size:
        return 0, sram_status

    # Attempt to allocate all tiles in the group using addr_decider
    temp_sram_status = sram_status.copy()
    remaining_loadable = loadable_amount
    tiles_loaded = []

    start_addr = temp_sram_status[-1][2] + 1 if temp_sram_status else 0
    for tile in tiles_to_load:
        if tile[1] > pcb_module.compute_module.core.SRAM_size:
            raise Exception("Tile size exceeds SRAM size")
        start_addr, end_addr = addr_decider(temp_sram_status, start_addr, tile[1], tile[1], pcb_module)
        if (start_addr, end_addr) == (-1, -1):
            return 0, sram_status
        temp_sram_status.append([tile[0], start_addr, end_addr, 1])
        temp_sram_status = sort_sram_status(temp_sram_status)
        remaining_loadable -= tile[1]
        tiles_loaded.append(tile)
        start_addr = end_addr + 1

    # If all tiles are allocated, update sram_status and data
    sram_status = temp_sram_status
    data = [tile for tile in data if tile not in tiles_loaded]

    with open(file_path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # 파일 읽기 잠금 (Shared Lock)
        json.dump(data, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)  # 잠금 해제

    return remaining_loadable, sram_status
'''
def get_sramutil(sram_list:list):
    sram_usage = 0
    for tile in sram_list:
        sram_usage += (tile[2] - tile[1])
    return sram_usage


def write_previous_ops_from_sram(
    sram_status: list,
    ops_name: str,
    loadable_amount: int
    ):
    ops_order = [
        'MHA',
        'q_projection',
        'k_projection',
        'v_projection',
        'q_mul_k',
        'softmax',
        'a_mul_v',
        'w0_projection',
        'FFN',
        'w1_projection',
        'gelu',
        'w2_projection'
    ]
    target_idx = 0

    ### Handling About Flash Attention ###
    use_flash_attention = False
    for i in range(len(sram_status)):
        if 'w0_projection' in ops_name and 'q_mul_k' in sram_status[i][0]:
            use_flash_attention = True
            break

    #### Handling End ####

    for i in range(len(ops_order)):
        if 'MHA' in ops_name:
            target_idx = 11         # w2_projection
        if ops_order[i] in ops_name:
            target_idx = i-1
            if 'w0_projection' in ops_name and use_flash_attention:
                target_idx = 4

    used_amount = 0
    tmp_sram_status = []
    for i in range(len(sram_status)):
        if ops_order[target_idx] not in sram_status[i][0]:
            tmp_sram_status.append(sram_status[i])
        elif 'alloc' in sram_status[i][0]:
            used_amount = sram_status[i][2] - sram_status[i][1] + 1

    remained_amount = loadable_amount - used_amount
    sram_status = tmp_sram_status

    return remained_amount, sram_status

def write_tile_from_sram(
    sram_status: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param: int,
    loadable_amount: int,
    ):

    #print("before write ", sram_status) # exist
    #print("write tile from sram")
    #print("ops_name", ops_name) #q projection
    #print("dec_param", dec_param) # 1
    #print("previous", previous_m_n_k) # _0_2_0

    previous_m_n_k = previous_m_n_k+"_"

    if 'a_mul_v' in ops_name:
        if len(ops_name)>8 and ops_name[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8','9']:
            ops_name = 'q_mul_k' + ops_name[7:]
    used_amount = 0
    tmp_sram_status = []
#    print("SKKIM sram status", ops_name, previous_m_n_k, dec_param,  sram_status)
    for i in range(len(sram_status)):
        if ops_name not in sram_status[i][0]:
            tmp_sram_status.append(sram_status[i])

        else:
            if 'load' in sram_status[i][0]:
                if dec_param == 2:
                    if 'load_M_N' not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])
                        
                    elif previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])
                        
                elif dec_param == 1:
                    if 'load_N_K' not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

                    elif previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

                elif dec_param == 0:
                    if 'load_M_K' not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

                    elif previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

            elif 'alloc' in sram_status[i][0]:
                if dec_param != -1:
                    if previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])
                    else:
                        used_amount = sram_status[i][2] - sram_status[i][1] + 1
    filtered_sram_status = []
    for i in tmp_sram_status:
        previous_mnk = previous_m_n_k.split("_")
        tile_mnk = i[0].split("_")
        #if 'load' in i[0] and ops_name in i[0] and  len(tile_mnk) > 3:
        if ops_name in i[0] and  len(tile_mnk) > 3:
            _, prv_m, prv_n, prv_k,_ = previous_mnk
            tile_m, tile_n, tile_k = tile_mnk[-4:-1]
            if 'softmax' in i[0]:
                if int(tile_m) < int(prv_m):
                    continue
                elif int(tile_m) == int(prv_m) and int(tile_n) < int(prv_n):
                    continue
            else:
                if int(tile_k) < int(prv_k):
                    continue
                elif int(tile_k) == int(prv_k) and int(tile_n) < int(prv_n):
                    continue
                elif int(tile_k) == int(prv_k) and int(tile_n) == int(prv_n) and int(tile_m) < int(prv_m):
                    continue
            filtered_sram_status.append(i)
        else:
            filtered_sram_status.append(i)
    tmp_sram_status = filtered_sram_status

    remained_amount = loadable_amount-used_amount
    if(remained_amount < 0):
        raise Exception('remained_amount is negative!')

    sram_status = tmp_sram_status

    sram_status = sort_sram_status(sram_status)
    
    return remained_amount, sram_status

def free_tile_from_sram(
    sram_status: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param = -1
    ):
#    print("free tile from sram", previous_m_n_k) #_0_0_0_
#    print("ops_name", ops_name) #a_mul_v
#    print("dec param", dec_param) #1
#    print("sram status", sram_status) 
    previous_m_n_k = previous_m_n_k+"_"
    tmp_sram_status = []
    for i in range(len(sram_status)):
        #print("check", sram_status[i][0])
        if ops_name not in sram_status[i][0]: #false
            tmp_sram_status.append(sram_status[i])
        else: #true
            if 'load' in sram_status[i][0]: # true
                if dec_param == 2:
                    if 'load_M_N' not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])
                        
                    elif previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])
                        
                elif dec_param == 1:
                    if 'load_N_K' not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

                    elif previous_m_n_k not in sram_status[i][0]:
                        tmp_sram_status.append(sram_status[i])

                elif dec_param == 0: #true
                    if 'load_M_K' not in sram_status[i][0]: # no
                        tmp_sram_status.append(sram_status[i])

                    elif previous_m_n_k not in sram_status[i][0]: 
                        tmp_sram_status.append(sram_status[i])

            elif 'alloc' in sram_status[i][0]:
                tmp_sram_status.append(sram_status[i])
    filtered_sram_status = []
    for i in tmp_sram_status:
        previous_mnk = previous_m_n_k.split("_")
        tile_mnk = i[0].split("_")
        if 'load' in i[0] and ops_name in i[0] and  len(tile_mnk) > 3:
            _, prv_m, prv_n, prv_k,_ = previous_mnk
            tile_m, tile_n, tile_k = tile_mnk[-4:-1]
            if 'softmax' in i[0]:
                if int(tile_m) < int(prv_m):
                    continue
                elif int(tile_n) < int(prv_n):
                    continue
            else:
                if int(tile_k) < int(prv_k):
                    continue
                elif int(tile_k) == int(prv_k) and int(tile_n) < int(prv_n):
                    continue
                elif int(tile_k) == int(prv_k) and int(tile_n) == int(prv_n) and int(tile_m) < int(prv_m):
                    continue

            filtered_sram_status.append(i)
        else:
            filtered_sram_status.append(i)
    tmp_sram_status = filtered_sram_status

    sram_status = tmp_sram_status
    sram_status = sort_sram_status(sram_status)
#    print("after free ", sram_status)
    
    return sram_status


def store_sram_status(sram_status):
    with open(f"./SRAM/sram_status_{process_id}.json", "w") as f:
        json.dump(sram_status, f, indent=2)

    return 0

def sort_sram_status(sram_status):
    sram_status = sorted(sram_status, key=lambda x: x[1]) 
    for i in range(len(sram_status)-1):
        if(sram_status[i][2] > sram_status[i+1][1]):
            print('sram_status : ', sram_status)
            print('sram_status[i][2] : ', sram_status[i][2])
            print('sram_status[i+1][1] : ', sram_status[i+1][1])
            raise Exception("Deduped Tile exists.")
    return sram_status


def check_needed_tile_loaded(
    sram_status: list,
    M: int,
    N: int,
    K: int,
    ops_name: str,
    ):
    #file_path = "./Tiles/whole_tile_list.json"
    process_dir = f"./Tiles"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"whole_tile_list_{process_id}.json")
    with open(file_path, 'r') as f:
        try:
            whole_tile = json.load(f)
        except json.JSONDecodeError:
            raise Exception("There are no tile file")

    is_loaded = False
    needed_tile = []
    for tile in whole_tile:
        if ops_name == 'gelu' or 'FFN' in ops_name or 'MHA' in ops_name:
            if ops_name in tile[0]:
                needed_tile.append(tile)
        else:
            if ops_name in tile[0] and '_' + str(M) + '_' + str(N) + '_' + str(K) + '_' in tile[0]:
                needed_tile.append(tile)
#    print('needed_tile in def', needed_tile)

    tmp_needed_tile = needed_tile
    needed_tile = []
    for tile in tmp_needed_tile:
        if(len(sram_status) == 0):
            needed_tile.append(tile)

        else:
#            print('############start sram status check###############')
            for i in range(len(sram_status)):
#                print(tile , sram_status[i][0] ,sram_status[i][3] == 1)
                if tile[0] == sram_status[i][0] and sram_status[i][3] == 1:
                    break
                if i == len(sram_status)-1:
                    needed_tile.append(tile)
 
    if needed_tile == []:
        is_loaded = True


#    print("final need tile:", needed_tile)
    return  is_loaded, needed_tile
def addr_decider(
    sram_status: list,
    start_addr: int,
    loadable_amount: int,
    tile_size: int,
    pcb_module: Device,
    algo = 'best_fit',#skkim
    ):
    alignment = 1
    global current_reg
    current_addr = 0
    sorted_status = sort_sram_status(sram_status)
    free_blocks = []

    # Find free memory blocks
    for tile in sorted_status:
        if current_addr < tile[1]:
            free_blocks.append([current_addr, tile[1] - 1])
        current_addr = max(current_addr, tile[2] + 1)
    if current_addr < pcb_module.compute_module.core.SRAM_size:
        free_blocks.append([current_addr, pcb_module.compute_module.core.SRAM_size - 1])

    # Align the start address
    def align_address(addr):
        if alignment == 1:
            return addr
        return ((addr + alignment - 1) // alignment) * alignment

    # Choose block based on allocation algorithm
    selected_block = None
    if algo == 'first_fit':
        for block in free_blocks:
            aligned_start = align_address(block[0])
            if aligned_start + tile_size - 1 <= block[1]:
                selected_block = [aligned_start, aligned_start + tile_size - 1]
                break
    elif algo == 'best_fit':
        min_slack = float('inf')
        for block in free_blocks:
            aligned_start = align_address(block[0])
            if aligned_start + tile_size - 1 <= block[1]:
                slack = block[1] - (aligned_start + tile_size - 1)
                if slack < min_slack:
                    min_slack = slack
                    selected_block = [aligned_start, aligned_start + tile_size - 1]
    elif algo == 'worst_fit':
        max_size = -1
        for block in free_blocks:
            aligned_start = align_address(block[0])
            if aligned_start + tile_size - 1 <= block[1]:
                block_size = block[1] - block[0] + 1
                if block_size > max_size:
                    max_size = block_size
                    selected_block = [aligned_start, aligned_start + tile_size - 1]
    elif algo == 'sequential':
        max_addr = pcb_module.compute_module.core.SRAM_size
        attempts = 0
        while attempts < 2:  # Try twice: first with current_addr, then with largest block
            aligned_start = align_address(current_reg)
            print("SKKIM DEBUG", aligned_start,tile_size, current_reg)
            if aligned_start + tile_size - 1 <= max_addr:
                # Check if the block is free
                is_free = True
                for tile in sorted_status:
                    tile_start, tile_end = tile[1], tile[2]
                    if not (aligned_start + tile_size - 1 < tile_start or aligned_start > tile_end):
                        is_free = False
                        break
                if is_free:
                    selected_block = [aligned_start, aligned_start + tile_size - 1]
                    # Update current_addr for next allocation
                    current_reg = aligned_start + tile_size
                    if current_reg > max_addr:
                        current_reg = 1  # Wrap around to 1
                    break
            # First attempt failed, set current_addr to largest free block
            attempts += 1
            if attempts == 1 and not selected_block:
                max_size = -1
                largest_block_start = 1
                for block in free_blocks:
                    block_size = block[1] - block[0] + 1
                    if block_size >= tile_size and block_size > max_size:
                        max_size = block_size
                        largest_block_start = block[0]
                current_reg = align_address(largest_block_start)
                if current_reg > max_addr:
                    current_reg = 1                     

    else:
        raise ValueError(f"Unknown allocation algorithm: {algo}")

    if selected_block:
        return selected_block[0], selected_block[1]
    return -1, -1



'''
def addr_decider(
    sram_status: list,
    start_addr: int,
    loadable_amount: int,
    tile_size: int,
    pcb_module: Device,
    ):
    end_addr = 0
    if start_addr + tile_size - 1 > pcb_module.compute_module.core.SRAM_size:
        window_start = 0
        for i in range(len(sram_status)):
            if sram_status[i][1] - window_start + 1 > tile_size:
                start_addr = window_start
                end_addr = start_addr + loadable_amount -1
                #print("SKKIM, tile overflow - find, ", start_addr, end_addr)
                break
            if i == len(sram_status) - 1:
                start_addr = -1
                end_addr = -1
            window_start = sram_status[i][2] + 1
    else:
        for sram_item in sram_status:
            if start_addr <= sram_item[1] and sram_item[1] <= start_addr + tile_size - 1:
                break
            if start_addr <= sram_item[2] and sram_item[2] <= start_addr + tile_size - 1:
                break

            if sram_item == sram_status[-1]:
                end_addr = start_addr + loadable_amount - 1
                return start_addr, end_addr
            
        window_start = 0
        for i in range(len(sram_status)):
            if sram_status[i][1] - window_start + 1 > tile_size:
                start_addr = window_start
                end_addr = start_addr + loadable_amount -1
                break
            if i == len(sram_status) - 1:
                start_addr = -1
                end_addr = -1
            window_start = sram_status[i][2] + 1

#    if start_addr == -1 and end_addr == -1:
#        print("SKKIM, tile overflow - find X")
#        print("SKKIM, ", sram_status)
    return start_addr, end_addr
'''

def flash_attention_write(
    sram_status: list,
    prev_ops_name : str,
    pcb_module: Device,
    ops_name: str,
    loadable_amount: int,
    ):

    tmp_sram_status = []
    
    used_amount = 0
    for item in sram_status:
        if prev_ops_name not in item[0]:
            tmp_sram_status.append(item)

        elif 'alloc' in item[0]:
            used_amount = item[2] - item[1] + 1


    sram_status = tmp_sram_status
    remained_amount = loadable_amount - used_amount

    return remained_amount, sram_status


def flashattention_check_needed_tile(
    sram_status: list,
    ops_name: str,
    ):
    #file_path = "./Tiles/whole_tile_list.json"
    process_dir = f"./Tiles"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"whole_tile_list_{process_id}.json")
    with open(file_path, 'r') as f:
        try:
            whole_tile = json.load(f)
        except json.JSONDecodeError:
            raise Exception("There are no tile file")

    is_loaded = False
    needed_tile = []
    ops_name = 'q_mul_k_' + ops_name[8:] + '_'

    for tile in whole_tile:
        if ops_name in tile[0]:
            needed_tile.append(tile)

    tmp_needed_tile = needed_tile
    needed_tile = []
    for tile in tmp_needed_tile:
        if(len(sram_status) == 0):
            needed_tile.append(tile)

        else:
            for i in range(len(sram_status)):
                if tile[0] == sram_status[i][0] and sram_status[i][3] == 1:
                    break
                if i == len(sram_status)-1:
                    needed_tile.append(tile)

    if needed_tile == []:
        is_loaded = True

    return  is_loaded, needed_tile


def print_sram():
    file_path = f"./SRAM/sram_status_{process_id}.json"
    with open(file_path, 'r') as f:
        try:
            sram_status = json.load(f)
        except json.JSONDecodeError:
            raise Exception("There are no SRAM json file")

    print(sram_status)


    return 0
