import os
import re

# =============================================================================
# Configuration
# =============================================================================
verilog_dir = '../../../data/energy/verilog'

# 파싱할 실제 파일 이름과 표에 출력할 짧은 이름 맵핑
hw_modules = {
    'find_zero': 'find_zero',
    'alloc': 'alloc',
    'address_check': 'addr_check',  # Table 양식에 맞춰 이름 단축
    'bt_lookup': 'bt_lookup',
    'free': 'free'
}

# =============================================================================
# Data Parsing
# =============================================================================
def parse_time_power(filepath):
    """Verilog .out 파일에서 time(ps)과 total_power 파싱"""
    stats = {'time_ps': 0.0, 'power': 0.0}
    if not os.path.exists(filepath):
        return stats
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Time 파싱 (data arrival time 값 추출. EDA 툴 디폴트 단위가 ps라고 가정)
    time_matches = re.findall(r'(\d+\.\d+)\s+data arrival time', content)
    if time_matches:
        stats['time_ps'] = float(time_matches[0])

    # 2. Power 파싱 (Total 행의 Total Power 추출)
    power_match = re.search(r'Total\s+[\d\.eE\+\-]+\s+[\d\.eE\+\-]+\s+[\d\.eE\+\-]+\s+([\d\.eE\+\-]+)\s+[\d\.]+%?', content)
    if power_match:
        # EDA 툴의 Power 단위가 표출 단위(pW)와 다를 경우 여기서 곱하기를 통해 스케일링 필요
        stats['power'] = float(power_match.group(1))

    return stats

# 데이터 수집
results = {}
for file_mod, print_mod in hw_modules.items():
    out_path = os.path.join(verilog_dir, f"{file_mod}.out")
    results[print_mod] = parse_time_power(out_path)

# =============================================================================
# Print Table (Horizontal Layout)
# =============================================================================
headers = ["Metric"] + list(hw_modules.values())

print("\n" + "="*85)
print(f"{'Latency and power consumption of each hardware module.':^85}")
print("="*85)

# 헤더 출력
header_str = f"{headers[0]:<12} |"
for h in headers[1:]:
    header_str += f" {h:>12} |"
print(header_str)
print("-" * 85)

# Time 행 출력
time_str = f"{'Time [ps]':<12} |"
for h in headers[1:]:
    val = results[h]['time_ps']
    time_str += f" {val:>12.1f} |"
print(time_str)

# Power 행 출력 (지수 표기법 적용, 예: 1.4e-01)
power_str = f"{'Power [pW]':<12} |"
for h in headers[1:]:
    val = results[h]['power']
    # x.xe-xx 포맷으로 변환하여 LaTeX의 x.x * 10^-x 폼을 흉내냄
    formatted_pow = f"{val:>12.1e}" 
    power_str += f" {formatted_pow} |"
print(power_str)

print("="*85 + "\n")
