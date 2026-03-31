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
    특정 조건을 만족하는 연산에 대해 하나의 블록만 SRAM에서 제거한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        sram_table (list): 현재 SRAM 블록 상태 (0 = free, 1 = allocated)
        previous_m_n_k (str): 이전 M, N, K 값
        pcb_module (Device): 하드웨어 모델
        ops_name (str): 현재 연산 이름
        dec_param (int): 결정 파라미터 (0, 1, 2)

    Returns:
        tuple: (list, list) -> (업데이트된 SRAM 상태, 업데이트된 SRAM 테이블)
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

            # 해당 조건을 만족하는 첫 블록 하나만 해제
            if tile[1]:  # block list가 비어있지 않은 경우
                #print("??????????",tile[1])
                block_to_free = tile[1].pop(0)  # 가장 앞의 block 하나만 제거
                sram_table[block_to_free] = 0  # 해당 block을 free 상태로 변경
                updated = True

                # block 리스트가 모두 비면, 해당 tile도 제거
                if not tile[1]:
                    sram_status.remove(tile)
                break  # 하나의 블록만 해제하고 종료

    if updated:
        sram_status = sort_sram_status(sram_status)

    return sram_status, sram_table

def update_longest_zero(sram_table: list):
    """가장 긴 연속 0 구간을 찾아 전역 변수에 저장."""
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

    # 마지막 연속 0 구간 확인
    if current_zero_length > max_zero_length:
        max_zero_length = current_zero_length
        max_zero_start = current_zero_start

    longest_zero_start = max_zero_start
    longest_zero_length = max_zero_length
    return 104  # find_zero 오버헤드

def addr_decider(sram_table: list, needed_blocks: int):
    """캐싱된 가장 긴 연속 0 구간을 사용하거나, 필요 시 새로 탐색."""
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

    # 캐싱된 구간 확인
    if longest_zero_start is not None and longest_zero_length >= needed_blocks:
        # 캐싱된 구간에서 필요한 블록 할당
        allocated_blocks = list(range(longest_zero_start, longest_zero_start + needed_blocks))
        # 캐시 업데이트
        longest_zero_start += needed_blocks
        longest_zero_length -= needed_blocks
    else:
        # 캐싱된 구간이 없거나 충분하지 않으면 새로 탐색
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

    if num_of_tile == 0: #SRAM에 어떤 타일도 없는 경우
        needed_blocks = ceil(data[0][1] / block_size)
        if 'alloc' in data[0][0]:
            if needed_blocks > total_blocks:
                raise Exception("Alloc tile size exceed SRAM size")

            allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
            if len(allocated_blocks) == 0:
                return 0, sram_status, sram_table, orig_loadable


            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
            sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
            for block in allocated_blocks:
                sram_table[block] = 1  # 블록 테이블 업데이트
            sram_status = sort_sram_status(sram_status)
            data[0][1] -= len(allocated_blocks) * block_size
            data[0][1] = max(data[0][1], 0)

            if status_flag == 1:
                data.pop(0)  # 전체 할당 완료된 경우 제거

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
                sram_status.append([data[0][0], allocated_blocks, status_flag])  # 일반 타일 추가
                for block in allocated_blocks:
                    sram_table[block] = 1  # 블록 테이블 업데이트
                sram_status = sort_sram_status(sram_status)
                #loadable_amount -= data[0][1]
                loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                data[0][1] -= len(allocated_blocks) * block_size
                data[0][1] = max(data[0][1], 0)

                if status_flag == 1:
                    data.pop(0)  # 전체 할당 완료된 경우 제거

                if len(data) == 0:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                if len(data) > 0 and 'alloc' in data[0][0]:  # 다음 타일이 alloc 타일인 경우
                    needed_blocks = ceil(data[0][1] / block_size)
                    allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                    tot_find_overhead += find_overhead 
                    if len(allocated_blocks) == 0:
                        loadable_amount = 0
                    else:
                        status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                        sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        for block in allocated_blocks:
                            sram_table[block] = 1  # 블록 테이블 업데이트
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


    else:
        for i in range(len(sram_status)):
            if sram_status[i][2] == 0: #SRAM에 transfer 중 끊긴 타일이 있다.(이 타일은 alloc tile은 아니다.)
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
                    sram_status[i][1].extend(allocated_blocks)  # 기존 블록 리스트에 추가
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
                            sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
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


                else:  # 일부만 할당 가능
                    partial_blocks, find_overhead = addr_decider(sram_table, loadable_amount // block_size)
                    tot_find_overhead += find_overhead 

                    if len(partial_blocks) == 0:
                        loadable_amount = 0
                    else:
                        sram_status[i][1].extend(partial_blocks)  # 기존 블록 리스트에 추가
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

    if num_of_tile == 0: #SRAM에 어떤 타일도 없는 경우
        needed_blocks = ceil(data[0][1] / block_size)
        if 'alloc' in data[0][0]:
            if needed_blocks > total_blocks:
                raise Exception("Alloc tile size exceed SRAM size")

            allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
            if len(allocated_blocks) == 0:
                return 0, sram_status, sram_table, orig_loadable


            status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
            sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
            for block in allocated_blocks:
                sram_table[block] = 1  # 블록 테이블 업데이트
            sram_status = sort_sram_status(sram_status)
            data[0][1] -= len(allocated_blocks) * block_size
            data[0][1] = max(data[0][1], 0)

            if status_flag == 1:
                data.pop(0)  # 전체 할당 완료된 경우 제거

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
                sram_status.append([data[0][0], allocated_blocks, status_flag])  # 일반 타일 추가
                for block in allocated_blocks:
                    sram_table[block] = 1  # 블록 테이블 업데이트
                sram_status = sort_sram_status(sram_status)
                #loadable_amount -= data[0][1]
                loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size, 0)
                data[0][1] -= len(allocated_blocks) * block_size
                data[0][1] = max(data[0][1], 0)

                if status_flag == 1:
                    data.pop(0)  # 전체 할당 완료된 경우 제거

                if len(data) == 0:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    #return 0, sram_status, sram_table, tot_find_overhead
                    return 0, sram_status, sram_table, orig_loadable


                if len(data) > 0 and 'alloc' in data[0][0]:  # 다음 타일이 alloc 타일인 경우
                    needed_blocks = ceil(data[0][1] / block_size)
                    allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                    tot_find_overhead += find_overhead 
                    if len(allocated_blocks) == 0:
                        loadable_amount = 0
                    else:
                        status_flag = 1 if len(allocated_blocks) == needed_blocks else 0
                        sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        for block in allocated_blocks:
                            sram_table[block] = 1  # 블록 테이블 업데이트
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


    else:
        for i in range(len(sram_status)):
            if sram_status[i][2] == 0: #SRAM에 transfer 중 끊긴 타일이 있다.(이 타일은 alloc tile은 아니다.)
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
                    sram_status[i][1].extend(allocated_blocks)  # 기존 블록 리스트에 추가
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
                            sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
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


                else:  # 일부만 할당 가능
                    partial_blocks, find_overhead = addr_decider(sram_table, loadable_amount // block_size)
                    tot_find_overhead += find_overhead 

                    if len(partial_blocks) == 0:
                        loadable_amount = 0
                    else:
                        sram_status[i][1].extend(partial_blocks)  # 기존 블록 리스트에 추가
                        for block in partial_blocks:
                            sram_table[block] = 1

                        data[0][1] -= len(partial_blocks)*block_size
                        data[0][1] = max(data[0][1], 0)
                        loadable_amount = 0


                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                break

            if i == len(sram_status)-1: #SRAM에 transfer가 중단된 타일이 없는 경우
                needed_blocks = ceil(data[0][1] / block_size)

                if 'alloc' in data[0][0]: #만약 다음 load해야할 타일이 alloc 타일인 경우
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
                        sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
                        data[0][1] -= len(allocated_blocks) * block_size
                        data[0][1] = max(data[0][1], 0)
                        #sram_status.append([data[0][0], allocated_blocks, 1])
                        for block in allocated_blocks:
                            sram_table[block] = 1
                        sram_status = sort_sram_status(sram_status)
                        if status_flag == 1: data.pop(0)

                else:
                    #print("load", data[0],loadable_amount)
                    if loadable_amount >= ceil(data[0][1] / block_size)*block_size: #loadable_amount가 다음 tile을 충분히 다 load할 수 있다면
                        allocated_blocks, find_overhead = addr_decider(sram_table, needed_blocks)
                        tot_find_overhead += find_overhead 
#                        print("skkim", allocated_blocks, sram_status, sram_table, needed_blocks)

                        if len(allocated_blocks) == 0:
                            loadable_amount = 0

                        elif len(allocated_blocks) < needed_blocks: #SRAM에 해당 타일을 쪼개어 넣어야 하는 경우,
                            sram_status.append([data[0][0], allocated_blocks, 0])
                            data[0][1] -= len(allocated_blocks) * block_size
                            data[0][1] = max(data[0][1], 0)
                            for block in allocated_blocks:
                                sram_table[block] = 1
                            sram_status = sort_sram_status(sram_status)
                            loadable_amount = max(loadable_amount - len(allocated_blocks) * block_size,0)


                        elif len (allocated_blocks) == needed_blocks: #SRAM에 해당 타일을 다 넣을 수 있는 자리가 있다면,
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
                                    sram_status.append([data[0][0], allocated_blocks, status_flag])  # Alloc 타일 추가
                                    data[0][1] -= len(allocated_blocks) * block_size
                                    data[0][1] = max(data[0][1], 0)
                                    #sram_status.append([data[0][0], allocated_blocks , 1])
                                    sram_status = sort_sram_status(sram_status)
                                    for block in allocated_blocks:
                                        sram_table[block] = 1
                                    if status_flag == 1: data.pop(0)
                        else:
                            assert(False) # allocate more tile
                    else: #loadable_amount가 다음 타일을 충분히 다 load할 수 없다면
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
    현재 사용 중인 SRAM 블록 개수를 계산한다.

    Parameters:
        sram_list (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)

    Returns:
        int: 사용 중인 SRAM 블록 개수
    """
    sram_usage = sum(len(tile[1]) for tile in sram_list)  # 모든 타일의 블록 개수를 합산
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
    이전 연산에 사용된 타일을 SRAM에서 제거하고, 사용된 메모리 양을 계산하며, SRAM 테이블을 업데이트한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        sram_table (list): 현재 SRAM 블록 상태 (0 = free, 1 = allocated)
        ops_name (str): 현재 연산 이름
        loadable_amount (int): 현재 가능한 SRAM 공간
        block_size (int): 블록 크기 (bytes)

    Returns:
        tuple: (int, list, list) -> (사용 후 남은 공간, 업데이트된 SRAM 상태, 업데이트된 SRAM 테이블)
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

    ### Flash Attention 처리 ###
    use_flash_attention = any(
        'w0_projection' in ops_name and 'q_mul_k' in item[0] for item in sram_status
    )

    ### 목표 연산 인덱스 찾기 ###
    for i, op in enumerate(ops_order):
        if 'MHA' in ops_name:
            target_idx = 11  # 'w2_projection'이 최종 연산
        elif op in ops_name:
            target_idx = i - 1
            if 'w0_projection' in ops_name and use_flash_attention:
                target_idx = 4  # Flash Attention 적용 시 'q_mul_k' 이후 연산까지 필요
    used_amount = 0
    tmp_sram_status = []

    # 불필요한 연산 제거 및 사용한 메모리 계산
    for tile in sram_status:
        if ops_order[target_idx] not in tile[0]:  # 이전 연산이 아닌 경우 유지
            tmp_sram_status.append(tile)
        elif 'alloc' in tile[0]:  # Alloc 타일인 경우 사용량 계산
            #print("DEBUG", tile)  # 블록 개수 기준 사용량 계산
            used_amount += len(tile[1])  # 블록 개수 기준 사용량 계산
            # SRAM 테이블에서 해당 블록 해제

    remained_amount = max(0, loadable_amount - (used_amount * block_size)) #skkim test
#    print("DEBUG ", remained_amount, loadable_amount , used_amount , block_size)
    sram_status = tmp_sram_status

    sram_table = [0] * len(sram_table)  # 모든 블록을 free 상태로 초기화
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1  # 남아있는 블록을 다시 할당 상태로 설정


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
    특정 연산에 사용된 타일을 SRAM에서 제거하고, 사용된 메모리 양을 계산한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        previous_m_n_k (str): 이전 M, N, K 값
        pcb_module (Device): 하드웨어 모델
        ops_name (str): 현재 연산 이름
        dec_param (int): 결정 파라미터 (0, 1, 2)
        loadable_amount (int): 현재 가능한 SRAM 공간

    Returns:
        tuple: (int, list) -> (사용 후 남은 공간, 업데이트된 SRAM 상태)
    """

    if 'a_mul_v' in ops_name and len(ops_name)>8 and ops_name[8] in '0123456789':
        ops_name = 'q_mul_k' + ops_name[7:]
    used_amount = 0
    tmp_sram_status = []


    # 불필요한 연산 제거 및 사용된 메모리 계산
    for tile in sram_status:
        if ops_name not in tile[0]:  # 연산이 다르면 유지
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
                    used_amount += len(tile[1])  # 블록 개수 기준 사용량 계산
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

    remained_amount = max(0, loadable_amount - (used_amount * block_size))  # 사용 후 남은 공간
    if remained_amount < 0:
        #print("DEBUG", loadable_amount, used_amount, block_size, remained_amount)
        raise Exception('remained_amount is negative!')

    sram_status = sort_sram_status(tmp_sram_status)

    sram_table = [0] * len(sram_table)  # 모든 블록을 free 상태로 초기화
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1  # 남아있는 블록을 다시 할당 상태로 설정
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
    SRAM에서 조건에 맞는 타일의 단 하나의 블록만 write 처리하고,
    사용한 메모리 크기만큼 loadable_amount에서 차감한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        sram_table (list): 현재 SRAM 블록 상태 (0 = free, 1 = allocated)
        previous_m_n_k (str): 이전 M, N, K 값
        pcb_module (Device): 하드웨어 모델
        ops_name (str): 현재 연산 이름
        dec_param (int): 결정 파라미터 (0, 1, 2)
        loadable_amount (int): 현재 가능한 SRAM 공간 (bytes)

    Returns:
        tuple: (남은 SRAM 공간, 업데이트된 SRAM 상태, 업데이트된 SRAM 테이블)
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
                # 블록 하나만 write 처리
                freed_block = tile[1].pop(0)
                sram_table[freed_block] = 0
                block_freed = True
                # 블록이 비었으면 tile 제거, 아니면 유지
                if tile[1]:
                    tmp_sram_status.append(tile)
            else:
                tmp_sram_status.append(tile)
        else:
            tmp_sram_status.append(tile)

    # 이전 M/N/K 기준으로 더 정교하게 정렬
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

    # 테이블 재정의
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
    특정 연산에 사용된 타일을 SRAM에서 제거하고, SRAM 테이블을 최신 sram_status를 기반으로 업데이트한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        sram_table (list): 현재 SRAM 블록 상태 (0 = free, 1 = allocated)
        previous_m_n_k (str): 이전 M, N, K 값
        pcb_module (Device): 하드웨어 모델
        ops_name (str): 현재 연산 이름
        dec_param (int): 결정 파라미터 (0, 1, 2)

    Returns:
        tuple: (list, list) -> (업데이트된 SRAM 상태, 업데이트된 SRAM 테이블)
    """

    tmp_sram_status = []

    # 불필요한 타일 제거
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
            elif 'alloc' in tile[0]:  # alloc된 타일은 유지
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

    # 최종 sram_status 정리
    sram_status = sort_sram_status(tmp_sram_status)

    # sram_status 기준으로 sram_table을 다시 세팅 (초기화 후 남아있는 블록만 다시 설정)
    sram_table = [0] * len(sram_table)  # 모든 블록을 free 상태로 초기화
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1  # 남아있는 블록을 다시 할당 상태로 설정

    return sram_status, sram_table


def store_sram_status(sram_status, sram_table):
    with open(f"./SRAM/sram_status_{process_id}.json", "w") as f:
        json.dump(sram_status, f, indent=2)
    with open(f"./SRAM/sram_table_{process_id}.json", "w") as f:
        json.dump(sram_table, f, indent=2)

    return 0

def sort_sram_status(sram_status):
    """
    SRAM 상태를 블록 인덱스 기준으로 정렬하고, 중복된 블록이 있는지 검사한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)

    Returns:
        list: 정렬된 SRAM 상태 리스트
    """
    # 블록 리스트의 첫 번째 블록 번호를 기준으로 정렬
#    sram_status = sorted(sram_status, key=lambda x: x[1][0] if x[1] else float('inf'))

    # 중복된 블록이 있는지 검사
    used_blocks = set()
    for tile in sram_status:
        for block in tile[1]:  # 할당된 블록 리스트 순회
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
 
    # 모든 타일이 로드되었는지 여부 확인
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
    Flash Attention을 위해 이전 연산의 타일을 SRAM에서 제거하고, 사용된 메모리 양을 계산하며, SRAM 테이블을 업데이트한다.

    Parameters:
        sram_status (list): 현재 SRAM 상태 (각 항목은 [타일 이름, 할당된 블록 리스트, 상태] 형태)
        sram_table (list): 현재 SRAM 블록 상태 (0 = free, 1 = allocated)
        prev_ops_name (str): 이전 연산 이름
        pcb_module (Device): 하드웨어 모델
        ops_name (str): 현재 연산 이름
        loadable_amount (int): 현재 가능한 SRAM 공간

    Returns:
        tuple: (int, list, list) -> (사용 후 남은 공간, 업데이트된 SRAM 상태, 업데이트된 SRAM 테이블)
    """

    tmp_sram_status = []
    used_amount = 0

    # 이전 연산(`prev_ops_name`)에서 사용된 `alloc` 타일 제거
    for item in sram_status:
        if prev_ops_name not in item[0]:  # 현재 연산이 아니면 유지
            tmp_sram_status.append(item)
        elif 'alloc' in item[0]:  # alloc된 타일인 경우
            used_amount += len(item[1])  # 블록 개수 기준 사용량 계산

    # 사용한 공간만큼 `remained_amount` 업데이트
    remained_amount = loadable_amount + (used_amount * block_size)

    # 최종 sram_status 정리
    sram_status = sort_sram_status(tmp_sram_status)

    # 📌 최종 `sram_status` 기준으로 `sram_table`을 재구성
    sram_table = [0] * len(sram_table)  # 모든 블록을 free 상태로 초기화
    for tile in sram_status:
        for block in tile[1]:
            sram_table[block] = 1  # 남아있는 블록을 다시 할당 상태로 설정

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
