import re
import statistics
import ast

def find_max_consecutive_in_list(addresses):
    if not addresses:
        return 0, []

    # 주소 리스트를 정렬 (필요한 경우 주석 해제)
    # addresses = sorted(addresses)
#    print(f"Input addresses: {addresses}")

    max_length = 1
    current_length = 1
    current_num = addresses[0]
    start_addr = current_num  # 현재 구간의 시작 주소
    addr_ranges = []  # (start, end) 튜플을 저장할 리스트

    for num in addresses[1:]:
        if num == current_num + 1:
            # 연속된 경우
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            # 연속이 끊긴 경우: 이전 구간의 (start, end)를 저장
            addr_ranges.append((start_addr, current_num))
            current_length = 1
            start_addr = num  # 새로운 구간의 시작 주소
        current_num = num

    # 마지막 구간 추가
    addr_ranges.append((start_addr, current_num))

    print(f"Address ranges: {addr_ranges}")
    print()
    print()
    print()
    return max_length, addr_ranges

def find_max_and_avg_consecutive_per_line(filename):
    try:
        # 파일 읽기
        with open(filename, 'r') as file:
            content = file.read()

        # 'sram status :'로 블록 분리
        blocks = content.split('sram status :')
        max_lengths = []

        # 각 블록 처리
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:  # 빈 블록 무시
                continue

            try:
                # 블록을 Python 리스트로 파싱
                sram_status = ast.literal_eval(block)
                
                # 리스트가 올바른 형식인지 확인
                if not isinstance(sram_status, list):
                    print(f"Block {i} is not a valid list: {block[:50]}...")
                    continue

                # 주소 리스트 수집
                print(sram_status)
                addresses = []
                for entry in sram_status:
                    if len(entry) < 2 or not isinstance(entry[1], list):
                        print(f"Invalid entry in block {i}: {entry}")
                        continue
                    addresses += entry[1]  # 주소 리스트 병합

                # 최대 연속 길이 계산
                if addresses:
                    line_max_length = find_max_consecutive_in_list(addresses)
                    max_lengths.append(line_max_length)
                else:
                    print(f"No valid addresses in block {i}")
                    max_lengths.append(0)

            except (SyntaxError, ValueError) as e:
                print(f"Error parsing block {i}: {e}, Block content: {block[:50]}...")
                max_lengths.append(0)  # 오류 발생 시 0 추가
                continue

        # 최대 및 평균 연속 길이 계산
        if not max_lengths:
            print("No valid blocks processed.")
            return 0, 0.0

        max_of_max_lengths = max(max_lengths)
        avg_of_max_lengths = statistics.mean(max_lengths)

        return max_of_max_lengths, avg_of_max_lengths

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return 0, 0.0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0.0

# 메인 실행
if __name__ == "__main__":
    filename = "tmp"
    max_len, avg_len = find_max_and_avg_consecutive_per_line(filename)
    print(f"Maximum consecutive address length across all lines: {max_len}")
    print(f"Average of maximum consecutive address lengths per line: {avg_len:.2f}")
