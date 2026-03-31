import sys
import csv
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ----------------------------------------------------------------------
# 1. Define model layers and name mappings
# ----------------------------------------------------------------------
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

MODEL_MAP = {
    "tiny_llama_quant": "TinyLLaMA (w4a8)", "tiny_llama": "TinyLLaMA (int8)",
    "gpt_neo_quant": "GPT-Neo (w4a8)", "gpt_neo": "GPT-Neo (int8)",
    "gpt_xl_quant": "GPT-3 XL (w4a8)", "gpt_xl": "GPT-3 XL (int8)",
    "gemma_2_2b_quant": "Gemma-2 (w4a8)", "gemma_2_2b": "Gemma-2 (int8)",
    "gpt_2.7b_quant": "GPT-3 2.7B (w4a8)", "gpt_2.7b": "GPT-3 2.7B (int8)",
    "llama2_quant": "LLaMA2 (w4a8)", "llama2": "LLaMA2 (int8)",
    "bloom_quant": "Bloom (w4a8)", "bloom": "Bloom (int8)",
    "gpt3_quant": "GPT-3 13B (w4a8)", "gpt3": "GPT-3 13B (int8)",
}

# ----------------------------------------------------------------------
# 2. Parsing and TTFT extraction function
# ----------------------------------------------------------------------
def get_ttft_from_file(filename, model_name):
    """Reads the file and returns the adjusted latency (TTFT) ONLY for seq_len = 1."""
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if len(row) >= 2:
                seq_len = int(row[0])
                # Hardcoded for TTFT: stop and return immediately when seq_len == 1
                if seq_len == 1:
                    original_latency = float(row[1])
                    layer_count = MODEL_LAYERS.get(model_name, 24)
                    return (original_latency / 24) * layer_count
    return 0.0

# ----------------------------------------------------------------------
# 3. Main execution smooth
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_ttft.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    
    directories = [
        os.path.join(base_dir, 'gemmini'),
        os.path.join(base_dir, 'capuchin'),
        os.path.join(base_dir, 'compiler-ideal'),
        os.path.join(base_dir, 'smooth'),
        os.path.join(base_dir, 'smooth-er')
    ]
    
    results = []

    # Collect TTFT data directly from raw .out files
    for directory in directories:
        if not os.path.exists(directory):
            continue

        policy_name = os.path.basename(directory)

        for filename in Path(directory).glob('*.out'):
            model_name = filename.stem
            try:
                ttft = get_ttft_from_file(filename, model_name)
                results.append({'model': model_name, 'policy': policy_name, 'TTFT': ttft})
            except Exception:
                pass

        # Handle missing files by using llama2.out as a fallback for bloom
        llama2_file = Path(directory) / 'llama2.out'
        if llama2_file.exists():
            for model_name in ['bloom']:
                try:
                    ttft = get_ttft_from_file(llama2_file, model_name)
                    results.append({'model': model_name, 'policy': policy_name, 'TTFT': ttft})
                except Exception:
                    pass
        
        # Handle missing files by using llama2_quant.out as a fallback for tiny_llama and bloom_quant
        llama2_quant_file = Path(directory) / 'llama2_quant.out'
        if llama2_quant_file.exists():
            for model_name in ['tiny_llama', 'bloom_quant']:
                try:
                    ttft = get_ttft_from_file(llama2_quant_file, model_name)
                    results.append({'model': model_name, 'policy': policy_name, 'TTFT': ttft})
                except Exception:
                    pass

    # Create DataFrame and map names
    df = pd.DataFrame(results)
    if df.empty:
        print("Error: No valid data found in the specified directory.")
        sys.exit(1)

    df["group"] = df["model"].map(MODEL_MAP)
    df = df.dropna(subset=['group'])  # Drop unmapped models

    # Normalize TTFT based on the 'compiler-ideal' policy
    metric = "TTFT"
    norm_values = {}
    for group_name, group_df in df.groupby("group"):
        try:
            base_series = group_df[group_df["policy"] == "compiler-ideal"][metric]
            if not base_series.empty:
                base = base_series.values[0]
                if base != 0:
                    norm_values.update({idx: row[metric] / base for idx, row in group_df.iterrows()})
                else:
                    norm_values.update({idx: 1.0 for idx, row in group_df.iterrows()})
            else:
                norm_values.update({idx: None for idx, row in group_df.iterrows()})
        except IndexError:
            norm_values.update({idx: None for idx, row in group_df.iterrows()})
            
    df["plot_value"] = df.index.map(lambda x: norm_values.get(x, None))
    df = df.dropna(subset=["plot_value"])

    # ----------------------------------------------------------------------
    # 4. Plot generation
    # ----------------------------------------------------------------------
    sns.set(style="white")
    color_map = {
        "compiler-ideal": "#D3D3D3", "gemmini": "#808080", "capuchin": "#A9A9A9",
        "smooth": "#404040", "smooth-er": "#66BECB"
    }
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 25,
        'axes.titlesize': 25, 'axes.labelsize': 25,
        'xtick.labelsize': 25, 'ytick.labelsize': 25,
        'legend.fontsize': 21, 'text.color': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    })
    
    x_order = list(MODEL_MAP.values())

    fig, ax = plt.subplots(figsize=(14, 2))

    sns.barplot(
        data=df, x="group", y="plot_value", hue="policy", order=x_order,
        hue_order=["compiler-ideal", "capuchin", "gemmini", "smooth", "smooth-er"],
        palette=color_map, edgecolor="black", ax=ax, legend=False
    )

    # Y-axis setup
    ax.set_yticks([0.5, 1.0])
    ax.set_ylim([0.5, 1.05])
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=3, zorder=2)
    ax.set_ylabel(f"Normalized\n{metric}")

    # X-axis setup
    ax.set_xticklabels(x_order, rotation=45, ha="right")
    ax.set_xlabel("")

    # Border style
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Legend setup
    legend_patches = [
        mpatches.Patch(facecolor=color_map["compiler-ideal"], edgecolor='black', label="Compiler-Ideal"),
        mpatches.Patch(facecolor=color_map["capuchin"], edgecolor='black', label="Capuchin"),
        mpatches.Patch(facecolor=color_map["gemmini"], edgecolor='black', label="Gemmini"),
        mpatches.Patch(facecolor=color_map["smooth"], edgecolor='black', label="SMOOTH-Base"),
        mpatches.Patch(facecolor=color_map["smooth-er"], edgecolor='black', label="SMOOTH-ER")
    ]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.3),
               ncol=5, frameon=False, edgecolor='black', columnspacing=1.0)

    # Setup output directory and save the figure
    output_path = f"{metric}_8MB.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')

    print(f"[Success] Data parsed and plot generated successfully -> {output_path}")

if __name__ == "__main__":
    main()
