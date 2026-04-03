from hardware_model.hw_device import Device
from math import ceil, log2, log
import json
import os
import multiprocessing
process_id = multiprocessing.current_process().pid
longest_zero_start = None
longest_zero_length = 0

def free_ended_block_from_sram(
    sram_status: list,
    sram_table: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param = -1
):
    """
    For operations that satisfy certain conditions, only one block is removed from SRAM.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        sram_table (list): Current SRAM block status (0 = free, 1 = allocated)
        previous_m_n_k (str): Previous M, N, K values
        pcb_module (Device): Hardware model
        ops_name (str): Current operation name
        dec_param (int): decision parameter (0, 1, 2)

    Returns:
        tuple: (list, list) -> (updated SRAM state, updated SRAM table)
    """
    previous_m_n_k += "_"
    updated = False

    for tile in sram_status:
        tile_name = tile[0]
        if ops_name in tile_name and 'load' in tile_name and previous_m_n_k in tile_name:
            if (dec_param == 2 and 'load_M_N' not in tile_name) or \
               (dec_param == 1 and 'load_N_K' not in tile_name) or \
               (dec_param == 0 and 'load_M_K' not in tile_name):
                continue

            # Release only the first block that satisfies the condition
            if tile[1]: # If block list is not empty
                #print("??????????",tile[1])
                block_to_free = tile[1].pop(0) # Remove only the first block
                sram_table[block_to_free] = 0 # Change the block to free state
                updated = True

                # When the block list is empty, the corresponding tile is also removed
                if not tile[1]:
                    sram_status.remove(tile)
                break # Release only one block and exit

    if updated:
        sram_status = sort_sram_status(sram_status)

    return sram_status, sram_table

def update_longest_zero(sram_table: list):
    """Find the longest consecutive zero interval and store it in a global variable."""
    global longest_zero_start, longest_zero_length
    max_zero_length = 0
    max_zero_start = 0
    current_zero_length = 0
    current_zero_start = 0

    for i, block in enumerate(sram_table):
        if block == 0:
            if current_zero_length == 0:
                current_zero_start = i
            current_zero_length += 1
            if current_zero_length > max_zero_length:
                max_zero_length = current_zero_length
                max_zero_start = current_zero_start
        else:
            current_zero_length = 0

    # Check the last consecutive 0 section
    if current_zero_length > max_zero_length:
        max_zero_length = current_zero_length
        max_zero_start = current_zero_start

    longest_zero_start = max_zero_start
    longest_zero_length = max_zero_length
    return 104 # find_zero overhead

def addr_decider(sram_table: list, needed_blocks: int):
    """Use the longest cached consecutive zeros, or search anew if necessary."""
    global longest_zero_start, longest_zero_length
    allocated_blocks = []
    find_zero_overhead = 0

    needed_blocks = int(needed_blocks)
    if needed_blocks < 0:
        print("Needed blocks < 0, ", needed_blocks)
        assert False
    if needed_blocks < 1:
        #return allocated_blocks
        return allocated_blocks, find_zero_overhead

    # Check cached section
    if longest_zero_start is not None and longest_zero_length >= needed_blocks:
        # Allocate necessary blocks in cached section
        allocated_blocks = list(range(longest_zero_start, longest_zero_start + needed_blocks))
        # update cache
        longest_zero_start += needed_blocks
        longest_zero_length -= needed_blocks
    else:
        # If there is no cached section or it is not sufficient, search again
        find_zero_overhead = update_longest_zero(sram_table)
        if longest_zero_length >= needed_blocks:
            allocated_blocks = list(range(longest_zero_start, longest_zero_start + needed_blocks))
            longest_zero_start += needed_blocks
            longest_zero_length -= needed_blocks

    #return allocated_blocks
#    if find_zero_overhead != 0:
#        print("FIND", find_zero_overhead)
    return allocated_blocks, find_zero_overhead


def load_sram_status(pcb_module):
#    block_size = model_dim
    block_size = pcb_module.compute_module.core.block_size
    total_sram_size = pcb_module.compute_module.core.SRAM_size

    sram_status = []
    sram_table = [0] * (total_sram_size // block_size)

    file_path = f"./SRAM/sram_status_{process_id}.json"
    table_path = f"./SRAM/sram_table_{process_id}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                sram_status = json.load(f)
            except json.JSONDecodeError:
                sram_status = []
    else:
        sram_status = []
    if os.path.exists(table_path):
        with open(table_path, 'r') as f:
            try:
                sram_table = json.load(f)
            except json.JSONDecodeError:
                sram_table = [0] * (total_sram_size // block_size)
    
    return sram_status, sram_table
def load_tile_to_sram_cont(
    sram_status: list,
    sram_table: list,
    pcb_module: Device,
    loadable_amount : int,
    ):
    block_size = pcb_module.compute_module.core.block_size
    orig_loadable = loadable_amount
    tot_find_overhead = 0
    
    file_path = f"./Tiles/remained_tile_list_{process_id}.json"
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            print("Json Error")
    
    num_of_tile = len(sram_status)
    total_blocks = len(sram_table)

    if len(data) == 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return 0, sram_status, sram_table, orig_loadable

    if num_of_tile == 0: #If there are no tiles in SRAM
        needed_blocks = ceil(data[0][1] / block_size)
        if 'alloc' in data[0][0]:
            if needed_blocks > total_blocks:
                raise Exception("Alloc tile size exceed SRAM size")

            allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
            if len(allocated_blocks) == 0:
                return 0, sram_status, sram_table, orig_loadable


            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
            sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
            for block in allocated_blocks:
                sram_table[block] = 1 # Update block table
            sram_status = sort_sram_status(sram_status)
            data[0][1] -= len(allocated_blocks) * block_size
            data[0][1] = max(data[0][1], 0)

            if status_flag == 1:
                data.pop(0) # Remove if all allocation is complete

        else:
            #if loadable_amount >= data[0][1]:
            if loadable_amount >= ceil(data[0][1] / block_size)*block_size:
                if needed_blocks > total_blocks:
                    raise Exception("Load tile size exceed SRAM size.")


                allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                tot_find_overhead += find_overhead 
                if len(allocated_blocks) == 0:
                    print(f"2. Not enough space for {data[0][0]}")
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add regular tiles
                for block in allocated_blocks:
                    sram_table[block] = 1 # Update block table
                sram_status = sort_sram_status(sram_status)
                #loadable_amount -= data[0][1]
                loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                data[0][1] -= len(allocated_blocks) * block_size
                data[0][1] = max(data[0][1], 0)

                if status_flag == 1:
                    data.pop(0) # Remove if all allocation is complete

                if len(data) == 0:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                if len(data) > 0 and 'alloc' in data[0][0]: # If the next tile is an alloc tile
                    needed_blocks = ceil(data[0][1] / block_size)
                    allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                    tot_find_overhead += find_overhead 
                    if len(allocated_blocks) == 0:
                        loadable_amount = 0
                    else:
                        status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                        sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        for block in allocated_blocks:
                            sram_table[block] = 1 # Update block table
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


    else:
        for i in range(len(sram_status)):
            if sram_status[i][2] == 0: #There is a tile in SRAM that was disconnected during transfer. (This tile is not an alloc tile.)
                needed_blocks = ceil(data[0][1] / block_size)
                allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                tot_find_overhead += find_overhead 
                if len(allocated_blocks) == 0:
#                    assert(False)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable



                if loadable_amount >= len(allocated_blocks) * block_size:
                    status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                    sram_status[i][2] = status_flag
                    sram_status[i][1].extend(allocated_blocks) # Add to existing block list
                    data[0][1] -= len(allocated_blocks) * block_size
                    data[0][1] = max(data[0][1], 0)
                    for block in allocated_blocks:
                        sram_table[block] = 1

                    loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                    if status_flag == 1:
                        data.pop(0)


                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        #return 0, sram_status, sram_table, tot_find_overhead
                        return 0, sram_status, sram_table, orig_loadable


                    if 'alloc' in data[0][0]:
                        needed_blocks = ceil(data[0][1] / block_size)
                        allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                        tot_find_overhead += find_overhead 

                        if len(allocated_blocks) == 0:
                            loadable_amount = 0
                        else:
                            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                            sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                            data[0][1] -= len(allocated_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            #sram_status.append([data[0][0], allocated_blocks, 1])
                            for block in allocated_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            if status_flag == 1: data.pop(0)

                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    json.dump(data, f, indent=2)
                                #return 0, sram_status, sram_table, tot_find_overhead
                                return 0, sram_status, sram_table, orig_loadable

                            #print("SRAM:", sram_status)


                else: # Only part of it can be assigned
                    partial_blocks, find_overhead = addr_decider(sram_table, loadable_amount // block_size)
                    tot_find_overhead += find_overhead 

                    if len(partial_blocks) == 0:
                        loadable_amount = 0
                    else:
                        sram_status[i][1].extend(partial_blocks) # Add to existing block list
                        for block in partial_blocks:
                            sram_table[block] = 1

                        data[0][1] -= len(partial_blocks)*block_size
                        data[0][1] = max(data[0][1], 0)
                        loadable_amount = 0


                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                break
            else:
                loadable_amount = 0

    return loadable_amount, sram_status, sram_table, orig_loadable



def load_tile_to_sram(
    sram_status: list,
    sram_table: list,
    pcb_module: Device,
    loadable_amount : int,
    ):
    block_size = pcb_module.compute_module.core.block_size
    orig_loadable = loadable_amount
    tot_find_overhead = 0
    
    file_path = f"./Tiles/remained_tile_list_{process_id}.json"
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            print("Json Error")
    
    num_of_tile = len(sram_status)
    total_blocks = len(sram_table)

    if len(data) == 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return 0, sram_status, sram_table, orig_loadable

    if num_of_tile == 0: #If there are no tiles in SRAM
        needed_blocks = ceil(data[0][1] / block_size)
        if 'alloc' in data[0][0]:
            if needed_blocks > total_blocks:
                raise Exception("Alloc tile size exceed SRAM size")

            allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
            if len(allocated_blocks) == 0:
                return 0, sram_status, sram_table, orig_loadable


            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
            sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
            for block in allocated_blocks:
                sram_table[block] = 1 # Update block table
            sram_status = sort_sram_status(sram_status)
            data[0][1] -= len(allocated_blocks) * block_size
            data[0][1] = max(data[0][1], 0)

            if status_flag == 1:
                data.pop(0) # Remove if all allocation is complete

        else:
            #if loadable_amount >= data[0][1]:
            if loadable_amount >= ceil(data[0][1] / block_size)*block_size:
                if needed_blocks > total_blocks:
                    raise Exception("Load tile size exceed SRAM size.")


                allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                tot_find_overhead += find_overhead 
                if len(allocated_blocks) == 0:
                    print(f"2. Not enough space for {data[0][0]}")
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add regular tiles
                for block in allocated_blocks:
                    sram_table[block] = 1 # Update block table
                sram_status = sort_sram_status(sram_status)
                #loadable_amount -= data[0][1]
                loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                data[0][1] -= len(allocated_blocks) * block_size
                data[0][1] = max(data[0][1], 0)

                if status_flag == 1:
                    data.pop(0) # Remove if all allocation is complete

                if len(data) == 0:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                if len(data) > 0 and 'alloc' in data[0][0]: # If the next tile is an alloc tile
                    needed_blocks = ceil(data[0][1] / block_size)
                    allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                    tot_find_overhead += find_overhead 
                    if len(allocated_blocks) == 0:
                        loadable_amount = 0
                    else:
                        status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                        sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        for block in allocated_blocks:
                            sram_table[block] = 1 # Update block table
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


    else:
        for i in range(len(sram_status)):
            if sram_status[i][2] == 0: #There is a tile in SRAM that was disconnected during transfer. (This tile is not an alloc tile.)
                needed_blocks = ceil(data[0][1] / block_size)
                allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                tot_find_overhead += find_overhead 
                if len(allocated_blocks) == 0:
#                    assert(False)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable



                if loadable_amount >= len(allocated_blocks) * block_size:
                    status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                    sram_status[i][2] = status_flag
                    sram_status[i][1].extend(allocated_blocks) # Add to existing block list
                    data[0][1] -= len(allocated_blocks) * block_size
                    data[0][1] = max(data[0][1], 0)
                    for block in allocated_blocks:
                        sram_table[block] = 1

                    loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                    if status_flag == 1:
                        data.pop(0)


                    if len(data) == 0:
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        #return 0, sram_status, sram_table, tot_find_overhead
                        return 0, sram_status, sram_table, orig_loadable


                    if 'alloc' in data[0][0]:
                        needed_blocks = ceil(data[0][1] / block_size)
                        allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                        tot_find_overhead += find_overhead 

                        if len(allocated_blocks) == 0:
                            loadable_amount = 0
                        else:
                            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                            sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                            data[0][1] -= len(allocated_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            #sram_status.append([data[0][0], allocated_blocks, 1])
                            for block in allocated_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            if status_flag == 1: data.pop(0)

                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    json.dump(data, f, indent=2)
                                #return 0, sram_status, sram_table, tot_find_overhead
                                return 0, sram_status, sram_table, orig_loadable

                            #print("SRAM:", sram_status)


                else: # Only part of it can be assigned
                    partial_blocks, find_overhead = addr_decider(sram_table, loadable_amount // block_size)
                    tot_find_overhead += find_overhead 

                    if len(partial_blocks) == 0:
                        loadable_amount = 0
                    else:
                        sram_status[i][1].extend(partial_blocks) # Add to existing block list
                        for block in partial_blocks:
                            sram_table[block] = 1

                        data[0][1] -= len(partial_blocks)*block_size
                        data[0][1] = max(data[0][1], 0)
                        loadable_amount = 0


                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                break

            if i == len(sram_status)-1: #If there are no tiles in SRAM where transfer has been interrupted
                needed_blocks = ceil(data[0][1] / block_size)

                if 'alloc' in data[0][0]: #If the next tile to load is an alloc tile
                    allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                    tot_find_overhead += find_overhead 

                    if len(allocated_blocks) == 0:
                        loadable_amount = 0
                        for i in range(len(sram_status)):
                            if 'alloc' in sram_status[i][0]: 
                                break
                            if i == len(sram_status) - 1:
                                print("Free All loaded Tile")
                                file_path = f"./Tiles/remained_tile_list_{process_id}.json"
                                with open(file_path, 'r') as f:
                                    try:
                                        data = json.load(f)
                                    except json.JSONDecodeError:
                                        data = []
                                        print("Json Decode Error")
                                for item in sram_status:
                                    data.insert(0, [item[0], len(item[1]) * block_size])
                                with open(file_path, 'w') as f:
                                    json.dump(data, f, indent = 2)

                                sram_status = []
                                sram_table = [0] * total_blocks 

                    else:
                        status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                        sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        #sram_status.append([data[0][0], allocated_blocks, 1])
                        for block in allocated_blocks:
                            sram_table[block] = 1
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

                else:
                    #print("load", data[0],loadable_amount)
                    if loadable_amount >= ceil(data[0][1] / block_size)*block_size: #if loadable_amount is enough to load the next tile
                        allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                        tot_find_overhead += find_overhead 
#                        print("skkim", allocated_blocks, sram_status, sram_table, needed_blocks)

                        if len(allocated_blocks) == 0:
                            loadable_amount = 0

                        elif len(allocated_blocks) < needed_blocks: #If the tile needs to be split into SRAM,
                            sram_status.append([data[0][0], allocated_blocks, 0])
                            data[0][1] -= len(allocated_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            for block in allocated_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size,0)


                        elif len (allocated_blocks) == needed_blocks: #If there is space in SRAM to fit all the tiles,
                            sram_status.append([data[0][0], allocated_blocks, 1])
                            data[0][1] -= len(allocated_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            for block in allocated_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size,0)
                            data.pop(0)
#                            print("load done.")
#                            if 'w0_projection' in data[0][0]:
#                                print(data[0],data[1],data[2],data[3])
                                #assert(False)
                            if len(data) == 0:
                                with open(file_path, 'w') as f:
                                    json.dump(data, f, indent=2)
                                #return 0, sram_status, sram_table, tot_find_overhead
                                return 0, sram_status, sram_table, orig_loadable


                            if 'alloc' in data[0][0]:
                                needed_blocks = ceil(data[0][1] / block_size)
                                allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                                tot_find_overhead += find_overhead 

                                if len(allocated_blocks) == 0:
                                    loadable_amount = 0

                                else:
                                    status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                                    sram_status.append([data[0][0], allocated_blocks, status_flag]) # Add Alloc tile
                                    data[0][1] -= len(allocated_blocks) * block_size
                                    data[0][1] = max(data[0][1], 0)
                                    #sram_status.append([data[0][0], allocated_blocks , 1])
                                    sram_status = sort_sram_status(sram_status)
                                    for block in allocated_blocks:
                                        sram_table[block] = 1
                                    if status_flag == 1: data.pop(0)
                        else:
                            assert(False) # allocate more tile
                    else: #loadable_amount is not enough to load the next tile
                        partial_blocks, find_overhead = addr_decider(sram_table, loadable_amount // block_size)
                        tot_find_overhead += find_overhead 
                        if len(partial_blocks) == 0:
                            loadable_amount = 0

                        else:
                            sram_status.append([data[0][0], partial_blocks, 0])
                            data[0][1] -= len(partial_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            for block in partial_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = 0


                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

    return loadable_amount, sram_status, sram_table, orig_loadable


def get_sramutil(sram_list: list,
    pcb_module: Device,
    ):
    block_size = pcb_module.compute_module.core.block_size
    """
    Calculate the number of SRAM blocks currently in use.

    Parameters:
        sram_list (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])

    Returns:
        int: Number of SRAM blocks in use
    """
    sram_usage = sum(len(tile[1]) for tile in sram_list) # Sum up the number of blocks of all tiles
    return sram_usage * block_size

def write_previous_ops_from_sram(
    sram_status: list,
    sram_table: list,
    ops_name: str,
    loadable_amount: int,
    pcb_module: Device,
    ):
    block_size = pcb_module.compute_module.core.block_size
    """
    Tiles used in previous operations are removed from SRAM, the amount of memory used is calculated, and the SRAM table is updated.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        sram_table (list): Current SRAM block status (0 = free, 1 = allocated)
        ops_name (str): Current operation name
        loadable_amount (int): Current available SRAM space
        block_size (int): Block size (bytes)

    Returns:
        tuple: (int, list, list) -> (space remaining after use, updated SRAM state, updated SRAM table)
    """
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

    ### Flash Attention Processing ###
    use_flash_attention = any(
        'w0_projection' in ops_name and 'q_mul_k' in item[0] for item in sram_status
    )

    ### Find target operation index ###
    for i, op in enumerate(ops_order):
        if 'MHA' in ops_name:
            target_idx = 11 # 'w2_projection' is the final operation
        elif op in ops_name:
            target_idx = i - 1
            if 'w0_projection' in ops_name and use_flash_attention:
                target_idx = 4 # When applying Flash Attention, calculations after 'q_mul_k' are required
    used_amount = 0
    tmp_sram_status = []

    # Eliminate unnecessary operations and calculate used memory
    for tile in sram_status:
        if ops_order[target_idx] not in tile[0]: # keep if not previous operation
            tmp_sram_status.append(tile)
        elif 'alloc' in tile[0]: # Calculate usage in case of Alloc tile
            used_amount += len(tile[1]) # Calculate usage based on number of blocks
            # Release that block from the SRAM table

    remained_amount = max(0, loadable_amount - (used_amount * block_size)) #skkim test
    sram_status = tmp_sram_status

    sram_table = [0] * len(sram_table) # Initialize all blocks to free state
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1 # Set remaining blocks to reallocate state


    return remained_amount, sram_status, sram_table

def write_tile_from_sram(
    sram_status: list,
    sram_table: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param: int,
    loadable_amount: int,
    ):
    previous_m_n_k = previous_m_n_k +"_"
    block_size = pcb_module.compute_module.core.block_size
    #print("write tile from sram", previous_m_n_k)
    #print("ops_name", ops_name) #a_mul_v
    #print("dec param", dec_param) #1
    #print("sram status", sram_status) 
    """
    Tiles used in a specific operation are removed from SRAM, and the amount of memory used is calculated.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        previous_m_n_k (str): Previous M, N, K values
        pcb_module (Device): Hardware model
        ops_name (str): Current operation name
        dec_param (int): decision parameter (0, 1, 2)
        loadable_amount (int): Current available SRAM space

    Returns:
        tuple: (int, list) -> (space remaining after use, updated SRAM status)
    """

    if 'a_mul_v' in ops_name and len(ops_name)>8 and ops_name[8] in '0123456789':
        ops_name = 'q_mul_k' + ops_name[7:]
    used_amount = 0
    tmp_sram_status = []


    # Eliminate unnecessary operations and calculate used memory
    for tile in sram_status:
        if ops_name not in tile[0]: # Maintain if operations are different
            tmp_sram_status.append(tile)
        else:
            if 'load' in tile[0]:
                if (dec_param == 2 and 'load_M_N' not in tile[0]) or \
                   (dec_param == 1 and 'load_N_K' not in tile[0]) or \
                   (dec_param == 0 and 'load_M_K' not in tile[0]) or \
                   (previous_m_n_k not in tile[0]):
                    tmp_sram_status.append(tile)

            elif 'alloc' in tile[0] and dec_param != -1:
                if previous_m_n_k not in tile[0]:
                    tmp_sram_status.append(tile)
                else:
                    used_amount += len(tile[1]) # Calculate usage based on number of blocks
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

    remained_amount = max(0, loadable_amount - (used_amount * block_size)) # Space remaining after use
    if remained_amount < 0:
        raise Exception('remained_amount is negative!')

    sram_status = sort_sram_status(tmp_sram_status)

    sram_table = [0] * len(sram_table) # Initialize all blocks to free state
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1 # Set remaining blocks to reallocate state
#    if check_sram_table != sram_table:
#        raise Exception('sram table error')

    return remained_amount, sram_status, sram_table

def write_ended_block_from_sram(
    sram_status: list,
    sram_table: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param: int,
    loadable_amount: int,
):
    """
    In SRAM, only one block of the tile that meets the conditions is written,
    The amount of memory used is deducted from loadable_amount.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        sram_table (list): Current SRAM block status (0 = free, 1 = allocated)
        previous_m_n_k (str): Previous M, N, K values
        pcb_module (Device): Hardware model
        ops_name (str): Current operation name
        dec_param (int): decision parameter (0, 1, 2)
        loadable_amount (int): Currently available SRAM space (bytes)

    Returns:
        tuple: (Remaining SRAM space, updated SRAM state, updated SRAM table)
    """
    previous_m_n_k += "_"
    block_size = pcb_module.compute_module.core.block_size

    if 'a_mul_v' in ops_name and len(ops_name) > 8 and ops_name[8] in '0123456789':
        ops_name = 'q_mul_k' + ops_name[7:]

    tmp_sram_status = []
    block_freed = False

    for tile in sram_status:
        tile_name = tile[0]

        if ops_name not in tile_name:
            tmp_sram_status.append(tile)
            continue

        if 'load' in tile_name:
            if (dec_param == 2 and 'load_M_N' not in tile_name) or \
               (dec_param == 1 and 'load_N_K' not in tile_name) or \
               (dec_param == 0 and 'load_M_K' not in tile_name) or \
               (previous_m_n_k not in tile_name):
                tmp_sram_status.append(tile)
                continue

        if 'alloc' in tile_name and dec_param != -1:
            if previous_m_n_k not in tile_name:
                tmp_sram_status.append(tile)
                continue

            if tile[1] and not block_freed:
                # Write only one block
                freed_block = tile[1].pop(0)
                sram_table[freed_block] = 0
                block_freed = True
                # If the block is empty, remove the tile, otherwise keep it
                if tile[1]:
                    tmp_sram_status.append(tile)
            else:
                tmp_sram_status.append(tile)
        else:
            tmp_sram_status.append(tile)

    # More precise sorting based on previous M/N/K
    previous_mnk = previous_m_n_k.split("_")
    filtered_status = []
    for i in tmp_sram_status:
        tile_mnk = i[0].split("_")
        if ops_name in i[0] and len(tile_mnk) > 3:
            _, prv_m, prv_n, prv_k, _ = previous_mnk
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
        filtered_status.append(i)

    sram_status = sort_sram_status(filtered_status)

    # Redefine table
    sram_table = [0] * len(sram_table)
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1

    remained_amount = max(0, loadable_amount - block_size) if block_freed else loadable_amount

    return remained_amount, sram_status, sram_table
    

def free_tile_from_sram(
    sram_status: list,
    sram_table: list,
    previous_m_n_k: str,
    pcb_module: Device,
    ops_name: str,
    dec_param = -1
    ):
    previous_m_n_k = previous_m_n_k + "_"
    #print("free tile from sram", previous_m_n_k) #_0_0_0_
    #print("ops_name", ops_name) #a_mul_v
    #print("dec param", dec_param) #1
    #print("sram status", sram_status) 

    """
    Tiles used in a specific operation are removed from SRAM, and the SRAM table is updated based on the latest sram_status.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        sram_table (list): Current SRAM block status (0 = free, 1 = allocated)
        previous_m_n_k (str): Previous M, N, K values
        pcb_module (Device): Hardware model
        ops_name (str): Current operation name
        dec_param (int): decision parameter (0, 1, 2)

    Returns:
        tuple: (list, list) -> (updated SRAM state, updated SRAM table)
    """

    tmp_sram_status = []

    # Remove unnecessary tiles
    for tile in sram_status: # tile[0] = a_mul_v_load_M_K_0_0_0_
        if ops_name not in tile[0]:  # false
            tmp_sram_status.append(tile)
        else: # true
            if 'load' in tile[0]:  # true
                if (dec_param == 2 and 'load_M_N' not in tile[0]) or \
                   (dec_param == 1 and 'load_N_K' not in tile[0]) or \
                   (dec_param == 0 and 'load_M_K' not in tile[0]) or \
                    previous_m_n_k not in tile[0]: #skkim todo
                    tmp_sram_status.append(tile)
            elif 'alloc' in tile[0]: # Allocated tiles are maintained
                tmp_sram_status.append(tile)

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
#    for i in tmp_sram_status:
#        print("(dec",dec_param,")",i[0], "is prv?", previous_m_n_k)

    # Final sram_status cleanup
    sram_status = sort_sram_status(tmp_sram_status)

    # Reset sram_table based on sram_status (reset only blocks remaining after initialization)
    sram_table = [0] * len(sram_table) # Initialize all blocks to free state
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1 # Set remaining blocks to reallocate state

    return sram_status, sram_table


def store_sram_status(sram_status, sram_table):
    with open(f"./SRAM/sram_status_{process_id}.json", "w") as f:
        json.dump(sram_status, f, indent=2)
    with open(f"./SRAM/sram_table_{process_id}.json", "w") as f:
        json.dump(sram_table, f, indent=2)

    return 0

def sort_sram_status(sram_status):
    """
    The SRAM state is sorted based on the block index and checked for duplicate blocks.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])

    Returns:
        list: Sorted SRAM status list
    """
    # Sort based on the first block number in the block list
#    sram_status = sorted(sram_status, key=lambda x: x[1][0] if x[1] else float('inf'))

    # Check if there are duplicate blocks
    used_blocks = set()
    for tile in sram_status:
        for block in tile[1]: # Traverse the allocated block list
            if block in used_blocks:
                print(f"Duplicate block detected: {block}")
                raise Exception("Deduped Tile exists.")
            used_blocks.add(block)

    return sram_status


def check_needed_tile_loaded(
    sram_status: list,
    M: int,
    N: int,
    K: int,
    ops_name: str,
    ):
    file_path = f"./Tiles/whole_tile_list_{process_id}.json"
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
                if tile[0] == sram_status[i][0] and sram_status[i][2] == 1:
                    break
                if i == len(sram_status)-1:
                    needed_tile.append(tile)
 
    # Check whether all tiles are loaded
    if len(needed_tile) == 0:
        is_loaded = True
    if is_loaded == True:
        lookup_len = []
        for tile in tmp_needed_tile:
            for i in range(len(sram_status)):
                if tile[0] == sram_status[i][0]:
                    lookup_len.append(len(sram_status[i][1]))
                    for cont in range(len(sram_status[i][1] ) - 1):
                        if sram_status[i][1] [cont+1] != sram_status[i][1] [cont] + 1:
                            lookup_len.append("FRAG")
        #print("LOOKUP",lookup_len)

#    print("final need tile:", needed_tile)
    return  is_loaded, needed_tile

def flash_attention_write(
    sram_status: list,
    sram_table: list,
    prev_ops_name: str,
    pcb_module: Device,
    ops_name: str,
    loadable_amount: int,
):
    block_size = pcb_module.compute_module.core.block_size
    """
    For Flash Attention, tiles from previous operations are removed from SRAM, the amount of used memory is calculated, and the SRAM table is updated.

    Parameters:
        sram_status (list): Current SRAM status (each item is in the form of [tile name, allocated block list, status])
        sram_table (list): Current SRAM block status (0 = free, 1 = allocated)
        prev_ops_name (str): Previous operation name
        pcb_module (Device): Hardware model
        ops_name (str): Current operation name
        loadable_amount (int): Current available SRAM space

    Returns:
        tuple: (int, list, list) -> (space remaining after use, updated SRAM state, updated SRAM table)
    """

    tmp_sram_status = []
    used_amount = 0

    # Remove `alloc` tile used in previous operation (`prev_ops_name`)
    for item in sram_status:
        if prev_ops_name not in item[0]: # If not the current operation, keep it
            tmp_sram_status.append(item)
        elif 'alloc' in item[0]: # In case of an allocated tile
            used_amount += len(item[1]) # Calculate usage based on number of blocks

    # Update `remained_amount` according to the space used
    remained_amount = loadable_amount + (used_amount * block_size)

    # Final sram_status cleanup
    sram_status = sort_sram_status(tmp_sram_status)

    # Reorganize `sram_table` based on the final `sram_status`
    sram_table = [0] * len(sram_table) # Initialize all blocks to free state
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1 # Set remaining blocks to reallocate state

    return remained_amount, sram_status, sram_table

def flashattention_check_needed_tile(
    sram_status: list,
    ops_name: str,
    ):
    file_path = f"./Tiles/whole_tile_list_{process_id}.json"
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
                if tile[0] == sram_status[i][0] and sram_status[i][2] == 1:
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
