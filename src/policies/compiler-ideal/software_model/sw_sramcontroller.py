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
    return sram_status
def load_tile_to_sram(
    sram_status: list,
    pcb_module: Device,
    loadable_amount : int,
    ):

    process_dir = f"./Tiles/"
    os.makedirs(process_dir, exist_ok=True)

    file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
    try:
        with open(file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN) # unlock
    except (json.JSONDecodeError, FileNotFoundError):
        data = []
    num_of_tile = len(sram_status)

    if len(data) == 0:
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN) # unlock
        return 0, sram_status


    if num_of_tile == 0: #If there are no tiles in SRAM
        start_addr = 0
        if 'alloc' in data[0][0]:
            if data[0][1] > pcb_module.compute_module.core.SRAM_size:
                raise Exception("Alloc tile size exceed SRAM size")

            else:
                end_addr = data[0][1] - 1
                sram_status.append([data[0][0], start_addr, end_addr, 1])
                sram_status = sort_sram_status(sram_status)
                data.pop(0)

        else:
            if loadable_amount >= data[0][1]:
                if data[0][1] > pcb_module.compute_module.core.SRAM_size:
                    raise Exception("Load tile size exceed SRAM size.")

                else:
                    end_addr = data[0][1] - 1
                    sram_status.append([data[0][0], start_addr, end_addr, 1])
                    sram_status = sort_sram_status(sram_status)
                    loadable_amount = loadable_amount - data[0][1]
                    data.pop(0)
                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                            json.dump(data, f, indent=2)
                            fcntl.flock(f, fcntl.LOCK_UN) # unlock
                        return 0, sram_status

                    if 'alloc' in data[0][0]:
                        start_addr = end_addr + 1
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else:
                            sram_status.append([data[0][0], start_addr, end_addr , 1])
                            sram_status = sort_sram_status(sram_status)
                            data.pop(0)


        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN) # unlock


    else:
        start_addr = sram_status[num_of_tile-1][2] + 1
        for i in range(len(sram_status)):
            if sram_status[i][3] == 0: #There is a tile that was disconnected during transfer in SRAM. (This tile is not an alloc tile.)
                if loadable_amount >= data[0][1]:
                    sram_status[i][3] = 1
                    sram_status[i][2] = sram_status[i][2] + data[0][1]
                    loadable_amount = loadable_amount - data[0][1]
                    data.pop(0)
                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                            json.dump(data, f, indent=2)
                            fcntl.flock(f, fcntl.LOCK_UN) # unlock
                        return 0, sram_status

                    if 'alloc' in data[0][0]:
                        start_addr = sram_status[i][2] + 1
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else:
                            sram_status.append([data[0][0], start_addr, end_addr, 1])
                            sram_status = sort_sram_status(sram_status)
                            data.pop(0)
                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                                    json.dump(data, f, indent=2)
                                    fcntl.flock(f, fcntl.LOCK_UN) # unlock
                                return 0, sram_status
                else:
                    org_loaded = sram_status[i][2]
                    sram_status[i][2] = sram_status[i][2] + loadable_amount
                    data[0][1] = data[0][1] - loadable_amount
                    loadable_amount = 0

                with open(file_path, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f, fcntl.LOCK_UN) # unlock

                break

            if i == len(sram_status)-1: #If there are no tiles in SRAM where transfer has been interrupted
                if 'alloc' in data[0][0]: #If the next tile to load is an alloc tile
                    start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                    if (start_addr, end_addr) == (-1, -1):
                        loadable_amount = 0
###
                        for i in range(len(sram_status)):
                            if 'alloc' in sram_status[i][0]: 
                                break
                            if i == len(sram_status) - 1:
                                process_dir = f"./Tiles/"
                                os.makedirs(process_dir, exist_ok=True)

                                file_path = os.path.join(process_dir, f"remained_tile_list_{process_id}.json")
                                with open(file_path, 'r') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                                    data = json.load(f)
                                    fcntl.flock(f, fcntl.LOCK_UN) # unlock
                                for item in sram_status:
                                    data.insert(0, [item[0], item[2] - item[1]])
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                                    json.dump(data, f, indent = 2)
                                    fcntl.flock(f, fcntl.LOCK_UN) # unlock
                                sram_status = []

                    else:
                        sram_status.append([data[0][0], start_addr, end_addr, 1])
                        sram_status = sort_sram_status(sram_status)
                        data.pop(0)


                else:
                    if loadable_amount >= data[0][1]: #if loadable_amount is enough to load the next tile
                        start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                        if (start_addr, end_addr) == (-1, -1):
                            loadable_amount = 0

                        else: #if there is space in SRAM to load the tiles,
                            sram_status.append([data[0][0], start_addr, end_addr, 1])
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = loadable_amount - data[0][1]
                            data.pop(0)
                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                                    json.dump(data, f, indent=2)
                                    fcntl.flock(f, fcntl.LOCK_UN) # unlock
                                return 0, sram_status

                            if 'alloc' in data[0][0]:
                                start_addr = end_addr + 1
                                start_addr, end_addr = addr_decider(sram_status, start_addr, data[0][1], data[0][1], pcb_module)

                                if (start_addr, end_addr) == (-1, -1):
                                    loadable_amount = 0

                                else:
                                    sram_status.append([data[0][0], start_addr, end_addr , 1])
                                    sram_status = sort_sram_status(sram_status)
                                    data.pop(0)
                    else: #loadable_amount is not enough to load the next tile
                        loadable_amount = 0

                with open(file_path, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_SH) # File read lock (Shared Lock)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f, fcntl.LOCK_UN) # unlock


    return loadable_amount, sram_status

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


    previous_m_n_k = previous_m_n_k+"_"

    if 'a_mul_v' in ops_name:
        if len(ops_name)>8 and ops_name[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8','9']:
            ops_name = 'q_mul_k' + ops_name[7:]
    used_amount = 0
    tmp_sram_status = []
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
    previous_m_n_k = previous_m_n_k+"_"
    tmp_sram_status = []
    for i in range(len(sram_status)):
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
