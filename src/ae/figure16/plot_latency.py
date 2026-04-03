import sys
import csv
import os
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Define layer counts for each model
MODEL_LAYERS = {
    'tiny_llama': 22, 'tiny_llama_quant': 22,
    'gpt_neo': 24, 'gpt_neo_quant': 24,
    'gpt_xl': 24, 'gpt_xl_quant': 24,
    'gemma_2_2b': 18, 'gemma_2_2b_quant': 18,
    'llama2': 32, 'llama2_quant': 32,
    'bloom': 30, 'bloom_quant': 30,
    'gpt_2.7b': 32, 'gpt_2.7b_quant': 32,
    'gpt3': 40, 'gpt3_quant': 40,
}

def parse_file_for_latency(filename, model_name):
    """
    Ignore unnecessary utilization and cycle data
    Only seq_len and latency required for the graph are parsed.
    The value of seq_len = 0 (Prompt Latency) is also read from here.
    """
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if len(row) >= 2: # Just need at least seq_len and original_latency
                seq_len = int(row[0])
                original_latency = float(row[1])
                layer_count = MODEL_LAYERS.get(model_name, 24)
                adjusted_latency = (original_latency / 24) * layer_count
                
                data.append({
                    'seq_len': seq_len,
                    'latency': adjusted_latency
                })
    return data

def collect_latencies(data, last_num):
    latencies = {}
    # Save from seq_len=0 (Prompt Latency)
    for n in range(0, last_num + 1):
        entry = next((e for e in data if e['seq_len'] == n), None)
        latencies[n] = entry['latency'] if entry else 0.0
    return latencies

def generate_csv(base_dir, last_num):
    # Mapping to actually existing paths
    directories = [
        os.path.join(base_dir, 'gemmini'),
        os.path.join(base_dir, 'capuchin'),
        os.path.join(base_dir, 'compiler-ideal'),
        os.path.join(base_dir, 'smooth'),
        os.path.join(base_dir, 'smooth-er')
    ]
    
    latency_file = 'latency_all.csv'
    latency_results = []

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' not found")
            continue

        for filename in Path(directory).glob('*.out'):
            model_name = filename.stem
            try:
                data = parse_file_for_latency(filename, model_name)
                latencies = collect_latencies(data, last_num)
                
                latency_row = [model_name, os.path.basename(directory)]
                # add seq_len_0
                for seq_len in range(0, last_num + 1):
                    latency_row.append(latencies.get(seq_len, 0.0))
                latency_results.append(latency_row)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        # Missing model handling: bloom borrows llama2 data
        llama2_file = Path(directory) / 'llama2.out'
        if llama2_file.exists():
            for model_name in ['bloom']:
                try:
                    data = parse_file_for_latency(llama2_file, model_name)
                    latencies = collect_latencies(data, last_num)
                    latency_row = [model_name, os.path.basename(directory)]
                    for seq_len in range(0, last_num + 1):
                        latency_row.append(latencies.get(seq_len, 0.0))
                    latency_results.append(latency_row)
                except Exception as e:
                    pass
        
        # Missing model handling: tiny_llama, bloom_quant borrow llama2_quant data
        llama2_quant_file = Path(directory) / 'llama2_quant.out'
        if llama2_quant_file.exists():
            for model_name in ['tiny_llama', 'bloom_quant']:
                try:
                    data = parse_file_for_latency(llama2_quant_file, model_name)
                    latencies = collect_latencies(data, last_num)
                    latency_row = [model_name, os.path.basename(directory)]
                    for seq_len in range(0, last_num + 1):
                        latency_row.append(latencies.get(seq_len, 0.0))
                    latency_results.append(latency_row)
                except Exception as e:
                    pass

    # Save the results as a CSV file
    with open(latency_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add seq_len_0 to header
        header = ['model', 'policy'] + [f'seq_len_{i}' for i in range(0, last_num + 1)]
        writer.writerow(header)
        for latency_row in latency_results:
            writer.writerow([latency_row[0], latency_row[1]] + [f"{x:.6f}" for x in latency_row[2:]])

    print(f"Latencies saved to {latency_file}")


def generate_plot(base_dir, max_seq_len):
    csv_path = "latency_all.csv"    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    seq_len_cols = [col for col in df.columns if col.startswith('seq_len_')]
    
    interval = 512

    # Filtering: seq_len_0 (Prompt), seq_len_1 (first token), and interval units
    target_cols = ['seq_len_0', 'seq_len_1'] + [
        col for col in seq_len_cols 
        if int(col.split('_')[-1]) % interval == 1 and int(col.split('_')[-1]) > 0
    ]
    # Remove and sort duplicates
    target_cols = sorted(list(set(target_cols)), key=lambda x: int(x.split('_')[-1]))

    df_long = df.melt(
        id_vars=["model", "policy"],
        value_vars=target_cols,
        var_name="seq_len",
        value_name="latency"
    )

    df_long["seq_len"] = df_long["seq_len"].str.replace('seq_len_', '').astype(int)
    df_long["latency"] = pd.to_numeric(df_long["latency"], errors="coerce")

    # 1. Separate Prompt Latency (seq_len == 0)
    prompt_df = df_long[df_long["seq_len"] == 0][["model", "policy", "latency"]].rename(columns={"latency": "prompt_latency"})
    
    # 2. Separate Generation Latency (seq_len > 0)
    df_long = df_long[df_long["seq_len"] > 0]

    all_seq_lens = range(1, max_seq_len + 1)
    models = df_long["model"].unique()
    policies = df_long["policy"].unique()

    all_combinations = pd.MultiIndex.from_product(
        [models, policies, all_seq_lens],
        names=['model', 'policy', 'seq_len']
    ).to_frame(index=False)

    df_expanded_merged = pd.merge(
        all_combinations,
        df_long,
        on=["model", "policy", "seq_len"],
        how="left"
    )

    df_expanded_merged['latency'] = df_expanded_merged.groupby(['model', 'policy'])['latency'].ffill()
    df_expanded = df_expanded_merged.dropna(subset=['latency'])

    # Policy Mapping
    policy_map = {
        "gemmini": "Gemmini",
        "compiler-ideal": "Compiler-Ideal",
        "capuchin": "Capuchin",
        "smooth": "SMOOTH-Base",
        "smooth-er": "SMOOTH-ER"
    }
    df_expanded["policy"] = df_expanded["policy"].map(policy_map)
    prompt_df["policy"] = prompt_df["policy"].map(policy_map)

    # Merge DataFrame (join prompt_latency based on mapped policy)
    df_expanded = pd.merge(df_expanded, prompt_df, on=["model", "policy"], how="left")
    df_expanded["prompt_latency"] = df_expanded["prompt_latency"].fillna(0.0)

    # 3. Accumulated latency + prompt latency
    df_expanded = df_expanded.sort_values(["model", "policy", "seq_len"])
    df_expanded["cumulative_latency"] = (df_expanded["prompt_latency"] + df_expanded.groupby(["model", "policy"])["latency"].cumsum()) / 60

    if df_expanded.empty:
        print("Error: No data to process.")
        sys.exit(1)

    model_display_map = {
        "tiny_llama_quant": "TinyLLaMA (w4a8)", "tiny_llama": "TinyLLaMA (int8)",
        "gpt_neo_quant": "GPT-Neo (w4a8)", "gpt_neo": "GPT-Neo (int8)",
        "gpt_xl_quant": "GPT-3 XL (w4a8)", "gpt_xl": "GPT-3 XL (int8)",
        "gemma_2_2b_quant": "Gemma-2 (w4a8)", "gemma_2_2b": "Gemma-2 (int8)",
        "gpt_2.7b_quant": "GPT-3 2.7B (w4a8)", "gpt_2.7b": "GPT-3 2.7B (int8)",
        "llama2_quant": "LLaMA2 (w4a8)", "llama2": "LLaMA2 (int8)",
        "bloom_quant": "Bloom (w4a8)", "bloom": "Bloom (int8)",
        "gpt3_quant": "GPT-3 13B (w4a8)", "gpt3": "GPT-3 13B (int8)"
    }
    df_expanded["model_display"] = df_expanded["model"].map(model_display_map)

    df_pivot = df_expanded.pivot_table(index=["model", "seq_len"], columns="policy", values="cumulative_latency").reset_index()
    df_pivot["vs_Compiler-Ideal"] = ((df_pivot["Compiler-Ideal"] - df_pivot["SMOOTH-ER"]) / df_pivot["Compiler-Ideal"] * 100).fillna(0)
    df_pivot["vs_Gemmini"] = ((df_pivot["Gemmini"] - df_pivot["SMOOTH-ER"]) / df_pivot["Gemmini"] * 100).fillna(0)
    df_pivot["vs_Capuchin"] = ((df_pivot["Capuchin"] - df_pivot["SMOOTH-ER"]) / df_pivot["Capuchin"] * 100).fillna(0)
    df_pivot["vs_SMOOTH-Base"] = ((df_pivot["SMOOTH-Base"] - df_pivot["SMOOTH-ER"]) / df_pivot["SMOOTH-Base"] * 100).fillna(0)

    print("\n" + "="*50)
    print("SMOOTH-ER Improvement Statistics (Overall)")
    print("-"*50)

    comparison_targets = ["Capuchin", "Compiler-Ideal", "Gemmini", "SMOOTH-Base"]
    for target in comparison_targets:
        col_name = f"vs_{target}"
        avg_val = df_pivot[col_name].mean()
        max_val = df_pivot[col_name].max()
        
        max_idx = df_pivot[col_name].idxmax()
        max_info = df_pivot.loc[max_idx]
        
        print(f"Vs {target:15}: Average = {avg_val:6.2f}%, Maximum = {max_val:6.2f}% "
              f"(Model: {max_info['model']}, Seq: {max_info['seq_len']})")
    print("="*50 + "\n")

    color_map = {
        "Gemmini": "#D3D3D3",
        "Compiler-Ideal": "#595959",
        "Capuchin": "#A9A9A9",
        "SMOOTH-Base": "#0D0D0D",
        "SMOOTH-ER": "#66BECB",
        "vs_Compiler-Ideal": "#595959",
        "vs_Gemmini": "#D3D3D3",
        "vs_Capuchin": "#A9A9A9" 
    }

    gain_legend_lines = [
        Patch(edgecolor='black', facecolor=color_map["vs_Capuchin"], label="Gain over Capuchin"),
        Patch(edgecolor='black', facecolor=color_map["vs_Compiler-Ideal"], label="Gain over Compiler-Ideal"),
        Patch(edgecolor='black', facecolor=color_map["vs_Gemmini"], label="Gain over Gemmini"),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='Gain by Prompt')
    ]

    base_legend_lines = [
        Line2D([0], [0], color=color_map["Capuchin"], linestyle='-', linewidth=2, label="Capuchin"),
        Line2D([0], [0], color=color_map["Compiler-Ideal"], linestyle='-', linewidth=2, label="Compiler-Ideal"),
        Line2D([0], [0], color=color_map["Gemmini"], linestyle='-', linewidth=2, label="Gemmini"),
        Line2D([0], [0], color=color_map["SMOOTH-Base"], linestyle='-', linewidth=2, label="SMOOTH-Base"),
        Line2D([0], [0], color=color_map["SMOOTH-ER"], linestyle='-', linewidth=2, label="SMOOTH-ER"),
    ]

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'text.color': 'black',
    })

    model_order = [
        "tiny_llama_quant", "gpt_neo_quant", "gpt_xl_quant", "gemma_2_2b_quant",
        "gpt_2.7b_quant", "llama2_quant", "bloom_quant", "gpt3_quant",
        "tiny_llama", "gpt_neo", "gpt_xl", "gemma_2_2b",
        "gpt_2.7b", "llama2", "bloom", "gpt3"
    ]

    fig, axes = plt.subplots(4, 4, figsize=(24, 7))
    axes_flat = axes.flatten()

    for i, model in enumerate(model_order):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        ax2 = ax.twinx()
        
        sub_df = df_expanded[df_expanded["model"] == model]
        sub_pivot = df_pivot[df_pivot["model"] == model]
        model_label = model_display_map.get(model, model)

        bar_seq_lens = [1, 8001, 16001, 24001, 32001]
        sub_pivot_bar = sub_pivot[sub_pivot["seq_len"].isin(bar_seq_lens)]
        
        total_bar_width = 7200
        w = total_bar_width / 3
        if not sub_pivot_bar.empty:
            def get_prompt_ratio(policy_name, current_seqs):
                # Retrieve the initial latency of that model/policy from prompt_df
                raw_l1_series = prompt_df.loc[(prompt_df["model"] == model) & (prompt_df["policy"] == policy_name), "prompt_latency"]
                raw_l1 = raw_l1_series.values[0] if len(raw_l1_series) > 0 else 0
                l1 = raw_l1 / 60.0 
                
                ratios = []
                for s in current_seqs:
                    ls_val = sub_df[(sub_df["policy"] == policy_name) & (sub_df["seq_len"] == s)]["cumulative_latency"].values
                    ls = ls_val[0] if len(ls_val) > 0 else 0
                    if ls == 0:
                        ratios.append(0)
                    else:
                        ratios.append(l1 / ls)
                return np.array(ratios)

            seq_vals = sub_pivot_bar["seq_len"].values
            
            gain_cap = sub_pivot_bar["vs_Capuchin"].fillna(0).values
            ratio_cap = get_prompt_ratio("Capuchin", seq_vals)
            ax.bar(seq_vals - w, gain_cap, width=w, edgecolor='black', color=color_map["vs_Capuchin"], linewidth=1)
            ax.bar(seq_vals - w, gain_cap * ratio_cap, width=w, edgecolor='black', facecolor='none', hatch='////', linewidth=0.5)

            gain_ci = sub_pivot_bar["vs_Compiler-Ideal"].fillna(0).values
            ratio_ci = get_prompt_ratio("Compiler-Ideal", seq_vals)
            ax.bar(seq_vals, gain_ci, width=w, edgecolor='black', color=color_map["vs_Compiler-Ideal"], linewidth=1)
            ax.bar(seq_vals, gain_ci * ratio_ci, width=w, edgecolor='black', facecolor='none', hatch='////', linewidth=0.5)

            gain_gem = sub_pivot_bar["vs_Gemmini"].fillna(0).values
            ratio_gem = get_prompt_ratio("Gemmini", seq_vals)
            ax.bar(seq_vals + w, gain_gem, width=w, edgecolor='black', color=color_map["vs_Gemmini"], linewidth=1)
            ax.bar(seq_vals + w, gain_gem * ratio_gem, width=w, edgecolor='black', facecolor='none', hatch='////', linewidth=0.5)

        for policy in ["Capuchin", "Compiler-Ideal", "Gemmini", "SMOOTH-Base", "SMOOTH-ER"]:
            p_df = sub_df[sub_df["policy"] == policy].sort_values("seq_len")
            if not p_df.empty:
                ax2.plot(p_df["seq_len"], p_df["cumulative_latency"], color=color_map[policy], marker='o', markersize=0.5, linewidth=1.2)

        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.set_xticks([1, 8000, 16000, 24000, 32000])
        ax2.set_xticklabels(["1", "8K", "16K", "24K", "32K"])

        ymax = sub_df["cumulative_latency"].max()
        ymax_val = int(math.ceil(ymax / 10) * 10) if not (math.isnan(ymax) or ymax == 0) else 100
        ax2.set_ylim(0, ymax if not (math.isnan(ymax) or ymax == 0) else 100)
        ax2.set_yticks(np.linspace(0, ymax_val, 3))

        ax.set_ylim(0, 80) 
        ax.set_yticks([0, 40, 80])

        ax.set_title(f"({chr(97+i)}) {model_label}", fontsize=20, y=-0.73)

    pos_row1 = axes[0, 0].get_position()
    pos_row2 = axes[1, 0].get_position()
    pos_row3 = axes[2, 0].get_position()
    pos_row4 = axes[3, 0].get_position()

    y_mid_12 = (pos_row1.y0 + pos_row1.y1 + pos_row2.y0 + pos_row2.y1) / 4
    y_mid_34 = (pos_row3.y0 + pos_row3.y1 + pos_row4.y0 + pos_row4.y1) / 4

    fig.text(0.08, y_mid_12, "Improvement [%]", va='center', ha='center', rotation='vertical', fontsize=18)
    fig.text(0.08, y_mid_34, "Improvement [%]", va='center', ha='center', rotation='vertical', fontsize=18)

    fig.text(0.925, y_mid_12, "Latency [min]", va='center', ha='center', rotation='vertical', fontsize=18)
    fig.text(0.925, y_mid_34, "Latency [min]", va='center', ha='center', rotation='vertical', fontsize=18)

    leg1 = fig.legend(handles=gain_legend_lines, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    leg2 = fig.legend(handles=base_legend_lines, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False)

    fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.8, wspace=0.19, left=0.1, right=0.9)

    output_path = 'latency_8MB.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    # Runtime parameters (remove seq_1_dir)
    base_dir = "../../../data/seq_32K/8MB/"
    last_num = 32768
    
    # When passed as an argument
    # Example: python script.py [base_dir] [last_num]
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        last_num = int(sys.argv[2])

    print(f"[*] Base directory set to: {base_dir}")
    print(f"[*] Max sequence length set to: {last_num}")

    print("\n[1/2] Generating latency_all.csv from .out files...")
    generate_csv(base_dir, last_num)

    print("\n[2/2] Generating Plot from latency_all.csv...")
    generate_plot(base_dir, last_num)
    
    print("\n[*] Execution Completed!")
