import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

# =============================================================================
# 1. Configuration & Parameters
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 17

sram_size = '8MB'
block_sizes = [256, 512, 1024, 2048, 4096]
block_str_map = {256: '256', 512: '512', 1024: '1K', 2048: '2K', 4096: '4K'} 

output_lengths = ['1K', '8K', '32K']
output_length_map = {'1K': 1025, '8K': 8193, '32K': 32257}

models = ['llama2_quant'] 
scale = 32 # LLaMA2 scale factor

policies = ['compiler-ideal', 'gemmini', 'capuchin', 'smooth-er']
policy_labels = {
    'compiler-ideal': 'Compiler-Ideal', 
    'gemmini': 'Gemmini', 
    'capuchin': 'Capuchin', 
    'smooth-er': 'SMOOTH-ER'
}

# 파일 경로 포맷
energy_out_dir = '../../../data/energy/energy_out'
seq_32k_dir = f'../../../data/seq_32K/{sram_size}'
verilog_dir = '../../../data/energy/verilog'

BASELINE_NPU_POWER_W = 4.0 

colors = {'compiler-ideal': '#D3D3D3', 'gemmini': '#A9A9A9', 'capuchin': '#808080', 'smooth-er': '#6BAED6', 'overhead': '#08306B'}
hatch_pattern = '//'

# =============================================================================
# 2. Data Parsing Functions
# =============================================================================
def parse_verilog_out(filepath):
    """Verilog .out 파일에서 time, leakage_pwr, total_pwr, area를 파싱"""
    stats = {'time': 0.0, 'total_pwr': 0.0, 'leakage_pwr': 0.0, 'area': 0.0}
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default zeros.")
        return stats
        
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Time 파싱 (data arrival time의 절대값 추출 후 초 단위 변환, ps -> s)
    time_matches = re.findall(r'(\d+\.\d+)\s+data arrival time', content)
    if time_matches:
        stats['time'] = float(time_matches[0]) * 1e-12

    # 2. Power 파싱 (Total 행에서 Leakage Power와 Total Power 추출)
    power_match = re.search(r'Total\s+[\d\.eE\+\-]+\s+[\d\.eE\+\-]+\s+([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)\s+[\d\.]+%?', content)
    if power_match:
        stats['leakage_pwr'] = float(power_match.group(1))
        stats['total_pwr'] = float(power_match.group(2))
        
    # 3. Area 파싱 (Chip area for module '\module_name': 값)
    area_match = re.search(r"Chip area for module\s+'[^']+':\s+([\d\.]+)", content)
    if area_match:
        stats['area'] = float(area_match.group(1))

    return stats

# 하드웨어 모듈 목록에 맞춰 동적으로 hw_stats 딕셔너리 생성
hw_modules = ['find_zero', 'alloc', 'address_check', 'bt_lookup', 'free']
hw_stats = {}
for mod in hw_modules:
    out_path = os.path.join(verilog_dir, f"{mod}.out")
    hw_stats[mod] = parse_verilog_out(out_path)

def get_policy_latency(policy, model_name, target_out_len):
    """기존 정책들의 Latency 파싱"""
    filepath = os.path.join(seq_32k_dir, policy, f"{model_name}.out")
    if not os.path.exists(filepath):
        return 0.0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[0].isdigit():
                if int(parts[0]) == target_out_len:
                    return float(parts[1])
    return 0.0

def get_block_latency(block_size_int, out_len_str, model_name):
    """SMOOTH(block)의 사이즈별 Latency 파싱"""
    bs_str = block_str_map[block_size_int]
    filepath = os.path.join(energy_out_dir, f"block{bs_str}_output{out_len_str}", f"{model_name}.out")
    if not os.path.exists(filepath):
        return 0.0
    with open(filepath, 'r') as f:
        for line in f:
            if "Latency" in line:
                parts = line.split(',')
                return float(parts[1].strip())
    return 0.0

def get_block_overhead_counts(block_size_int, out_len_str, model_name):
    """SMOOTH의 Overhead 호출 횟수 파싱"""
    bs_str = block_str_map[block_size_int]
    filepath = os.path.join(energy_out_dir, f"block{bs_str}_output{out_len_str}", f"{model_name}_overhead.out")
    find_zero_cnt, frag_cnt, address_check_sum = 0, 0, 0
    if not os.path.exists(filepath):
        return find_zero_cnt, frag_cnt, address_check_sum
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if "SKKIM OVERHEAD FIND ZERO" in line:
                find_zero_cnt += 1
            elif "SKKIM OVERHEAD LOOKUP" in line:
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    items = match.group(1).split()
                    frag_cnt += items.count("'FRAG'")
                    for item in items:
                        if item != 'FRAG' and item.replace('.', '').isdigit():
                            address_check_sum += float(item)
    return find_zero_cnt, frag_cnt, address_check_sum

def calculate_overhead_energy(find_zero, frag, addr_check, total_latency):
    """오버헤드 모듈의 에너지 소모량 계산 (J 단위로 변환하여 반환)"""
    PW_TO_W = 1e-12
    # parse_verilog_out 에서 받아온 total_pwr와 leakage_pwr를 사용
    dyn_pwr = {k: (v['total_pwr'] - v['leakage_pwr']) * PW_TO_W for k, v in hw_stats.items()}
    counts = {
        'find_zero': find_zero,
        'alloc': find_zero,
        'free': find_zero,
        'bt_lookup': frag,
        'address_check': addr_check
    }
    
    dyn_energy = 0.0
    for k in counts:
        dyn_energy += counts[k] * hw_stats[k]['time'] * dyn_pwr[k]
        
    static_energy = sum(hw_stats[k]['leakage_pwr'] for k in hw_stats) * total_latency
    
    return (dyn_energy + static_energy) 

# =============================================================================
# 3. Data Collection
# =============================================================================
model = models[0] 
energy_data = {out_len: {pol: 0.0 for pol in ['compiler-ideal', 'gemmini', 'capuchin']} for out_len in output_lengths}
for out_len in output_lengths:
    energy_data[out_len]['smooth-er'] = []

overhead_energy_data = {out_len: [] for out_len in output_lengths}

for out_len in output_lengths:
    target_len = output_length_map[out_len]
    
    # 1. 기존 정책 에너지 단일 값 계산 (J)
    for pol in ['compiler-ideal', 'gemmini', 'capuchin']:
        lat = get_policy_latency(pol, model, target_len)
        energy_data[out_len][pol] = lat * BASELINE_NPU_POWER_W

    # 2. SMOOTH (Block) 사이즈별 에너지 계산 (J)
    for bs in block_sizes:
        lat = get_block_latency(bs, out_len, model)
        base_energy_J = lat * BASELINE_NPU_POWER_W
        
        f_zero, frag, a_chk = get_block_overhead_counts(bs, out_len, model)
        
        ov_energy = (calculate_overhead_energy(f_zero * scale, frag * scale, a_chk * scale, lat)) 
        
        energy_data[out_len]['smooth-er'].append(base_energy_J)
        overhead_energy_data[out_len].append(ov_energy)

# =============================================================================
# 4. Plotting
# =============================================================================
fig = plt.figure(figsize=(14, 2))
gs = gridspec.GridSpec(1, len(output_lengths), wspace=0.21)

x_baselines = [0, 1, 2]
x_blocks = [4, 5, 6, 7, 8]
x_ticks = x_baselines + x_blocks
x_ticklabels = ['Compiler-\n    Ideal', 'Gemmini', 'Capuchin', '256B', '512B', '1KB', '2KB', '4KB']
bar_width = 0.7
axes = []

for i, out_len in enumerate(output_lengths):
    ax = fig.add_subplot(gs[0, i])
    axes.append(ax)

    # Baseline 정책들
    ax.bar(x_baselines[0], energy_data[out_len]['compiler-ideal'], width=bar_width, label=policy_labels['compiler-ideal'], color=colors['compiler-ideal'], edgecolor='black')
    ax.bar(x_baselines[1], energy_data[out_len]['gemmini'], width=bar_width, label=policy_labels['gemmini'], color=colors['gemmini'], edgecolor='black')
    ax.bar(x_baselines[2], energy_data[out_len]['capuchin'], width=bar_width, label=policy_labels['capuchin'], color=colors['capuchin'], edgecolor='black')
    
    # SMOOTH (블록 사이즈별)
    ax.bar(x_blocks, energy_data[out_len]['smooth-er'], width=bar_width, label=policy_labels['smooth-er'], color=colors['smooth-er'], edgecolor='black')
    
    # Overhead 스택 (J 단위 적용)
    ax.bar(x_blocks, overhead_energy_data[out_len], width=bar_width, bottom=energy_data[out_len]['smooth-er'], 
           label='Module Overhead', color=colors['overhead'], hatch=hatch_pattern, edgecolor='black')
    
    ax.set_title(f'({chr(97+i)}) {out_len}-th Token', y=-1.0)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, rotation=90)
    
    # Y축 단위 J에 맞춰 범위 및 틱 수정
    if i == 0: # (a) 1K Token
        ax.set_ylim(0, 0.4)
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_ylabel('Energy (J)')

    elif i == 1: # (b) 8K Token
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.3, 0.6])

    elif i == 2: # (c) 32K Token
        ax.set_ylim(0, 1.4)
        ax.set_yticks([0, 0.7, 1.4])

    # ==========================================
    # 1. 첫 번째 돋보기 창: 512B 단독 확대
    # ==========================================
    axins1 = ax.inset_axes([0.45, 0.48, 0.23, 0.38]) 
    
    axins1.bar(x_blocks, energy_data[out_len]['smooth-er'], width=bar_width, color=colors['smooth-er'], edgecolor='black')
    axins1.bar(x_blocks, overhead_energy_data[out_len], width=bar_width, bottom=energy_data[out_len]['smooth-er'], 
               color=colors['overhead'], hatch=hatch_pattern, edgecolor='black')
    
    target_idx_1 = 1 # 512B 인덱스
    min_base_1 = energy_data[out_len]['smooth-er'][target_idx_1]
    max_total_1 = energy_data[out_len]['smooth-er'][target_idx_1] + overhead_energy_data[out_len][target_idx_1]
    max_overhead_1 = overhead_energy_data[out_len][target_idx_1]
    
    # 텍스트 상단 여백 확보
    margin_y_1 = max_overhead_1 * 2.5 if max_overhead_1 > 0 else 1e-8
    axins1.set_ylim(min_base_1 - (margin_y_1 * 0.2), max_total_1 + margin_y_1)
    
    axins1.set_xlim(x_blocks[target_idx_1] - (bar_width / 2) - 0.1, x_blocks[target_idx_1] + (bar_width / 2) + 0.1)
    axins1.set_xticks([])
    axins1.set_yticks([])

    ov_val_1_uJ = max_overhead_1 * 1e9 # J 단위를 nanoJ (10^9) 단위로 변환
    axins1.text(x_blocks[target_idx_1], max_total_1 + (margin_y_1 * 0.1), f"{ov_val_1_uJ:.1f} nJ", 
                ha='center', va='bottom', fontsize=16, fontweight='bold', color=colors['overhead'])

    # ==========================================
    # 2. 두 번째 돋보기 창: 1KB 단독 확대
    # ==========================================
    axins2 = ax.inset_axes([0.72, 0.48, 0.23, 0.38]) 
    
    axins2.bar(x_blocks, energy_data[out_len]['smooth-er'], width=bar_width, color=colors['smooth-er'], edgecolor='black')
    axins2.bar(x_blocks, overhead_energy_data[out_len], width=bar_width, bottom=energy_data[out_len]['smooth-er'], 
               color=colors['overhead'], hatch=hatch_pattern, edgecolor='black')
              
    target_idx_2 = 2 # 1KB 인덱스
    min_base_2 = energy_data[out_len]['smooth-er'][target_idx_2]
    max_total_2 = energy_data[out_len]['smooth-er'][target_idx_2] + overhead_energy_data[out_len][target_idx_2]
    max_overhead_2 = overhead_energy_data[out_len][target_idx_2]
    
    margin_y_2 = max_overhead_2 * 2.5 if max_overhead_2 > 0 else 1e-8
    axins2.set_ylim(min_base_2 - (margin_y_2 * 0.2), max_total_2 + margin_y_2)
    
    axins2.set_xlim(x_blocks[target_idx_2] - (bar_width / 2) - 0.1, x_blocks[target_idx_2] + (bar_width / 2) + 0.1)
    axins2.set_xticks([])
    axins2.set_yticks([])

    ov_val_2_uJ = max_overhead_2 * 1e9 
    axins2.text(x_blocks[target_idx_2], max_total_2 + (margin_y_2 * 0.1), f"{ov_val_2_uJ:.1f} nJ", 
                ha='center', va='bottom', fontsize=16, fontweight='bold', color=colors['overhead'])

    # 확대 선 연결
    mark_inset(ax, axins1, loc1=3, loc2=4, fc="none", ec="0.5") 
    mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5")        

    if i == 0:
        ax.set_ylabel('Energy (J)')

    ax.axvline(x=3, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)

plt.tight_layout()
plt.savefig('energy_8MB.eps', format='eps', dpi=300, bbox_inches='tight')
print("Energy comparison plot saved as: energy_8MB.eps")

# =============================================================================
# 5. Print Experimental Results (Energy)
# =============================================================================
print("\n" + "="*60)
print(" Experimental Energy Results (Joules)")
print("="*60)

for out_len in output_lengths:
    print(f"\n[ Output Length: {out_len} Token ]")
    
    for pol in ['compiler-ideal', 'gemmini', 'capuchin']:
        val = energy_data[out_len][pol]
        print(f"  - {policy_labels[pol]:<16}: {val:.6f} J")
        
    print(f"  - {policy_labels['smooth-er']} (Base Energy + Overhead = Total):")
    for idx, bs in enumerate(block_sizes):
        base_nrg = energy_data[out_len]['smooth-er'][idx]
        ov_nrg = overhead_energy_data[out_len][idx]
        total_nrg = base_nrg + ov_nrg
        print(f"      {block_str_map[bs]:>4}B : {base_nrg:.6f} J  +  {ov_nrg:.2e} J  =  {total_nrg:.6f} J")

# =============================================================================
# 6. Print Area Overhead Results (Table Format)
# =============================================================================
print("\n" + "="*44)
print(" Hardware Module Area Overhead")
print("="*44)
print(f"| {'Module Name':<18} | {'Area (um^2)':<17} |")
print("-" * 44)

total_area = 0.0
for mod in hw_modules:
    area = hw_stats[mod]['area']
    total_area += area
    print(f"| {mod:<18} | {area:>17.6f} |")

print("-" * 44)
print(f"| {'Total Overhead':<18} | {total_area:>17.6f} |")
print("="*44 + "\n")
