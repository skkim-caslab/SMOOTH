import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "fa_nopreload_parse.csv"
test_fa_flag = True
save_png = True

df = pd.read_csv(file_path)
# Replace all -1 values with 0
for col in df.columns:
    if col != 'operation':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df.replace(-1, 0)
df = df.replace('-1', 0)
#df.to_csv(file_path, index=False)
print(f"Updated CSV file saved to {file_path}")

op_order = ['layernorm(MHA)', 'q projection', 'k projection', 'v projection', 'q_mul_k', 'softmax', 'a_mul_v',
            'w0_projection', 'layernorm(FFN)', 'w1_projection', 'gelu', 'w2_projection']
if "fa" in file_path:
    op_order = ['layernorm(MHA)', 'q projection(flash attention)', 'k projection(flash attention)', 'v projection(flash attention)', 'flash attention', 
                'w0_projection', 'layernorm(FFN)', 'w1_projection', 'gelu', 'w2_projection']
if test_fa_flag == True:
    op_order = ['flash attention']


#op_order = ['q projection']  # Modify this list for the specific operations you want to plot.
tot_cycle = []
fa_flag = -1 #qkt - 0, softmax - 1, sv - 2 
for op_name in op_order:
    tmp_df = df[df['operation'].isin([op_name])]
    total_cycle_max = int(max(tmp_df['total cycle']))
    tot_cycle.append(total_cycle_max)
    result = {
        'timetick': range(total_cycle_max),
        'memory bw': [0] * total_cycle_max,
        'sram occupancy': [0] * total_cycle_max,
        'sa util': [0] * total_cycle_max,
        'va util': [0] * total_cycle_max
    }
    start_session = 0
    for _, row in tmp_df.iterrows():
        if 'flash attention' == op_name:
            fa_flag = (fa_flag+1)%3
        # memory bw 계산

        if fa_flag == 2: ##a mul v
            start_mem_bw = start_session + int(row['compute cycle'])
            end_compute = start_mem_bw
            end_mem_bw = start_mem_bw + int(row['io cycle'])
            for timetick in range(start_mem_bw, min(end_mem_bw, total_cycle_max)):
                result['memory bw'][timetick] = row['memory bw']
            # sram occupancy 계산
            for timetick in range(start_session, min(end_compute, total_cycle_max)):
                result['sram occupancy'][timetick] = row['sram occupancy']
                result['sa util'][timetick] = row['sa util']
                result['va util'][timetick] = row['va util']
#            if op_name == 'flash attention':
#                print("FA,", fa_flag,"mem bw - ", start_mem_bw, end_mem_bw)
#                print("FA,", fa_flag,"compute - ", start_session, end_compute)
        else:
            end_mem_bw = start_session + int(row['io cycle'])
            end_session = start_session + int(row['current cycle'])
            for timetick in range(start_session, min(end_mem_bw, total_cycle_max)):
                result['memory bw'][timetick] = row['memory bw']
            # sram occupancy 계산
            for timetick in range(start_session, min(end_session, total_cycle_max)):
                result['sram occupancy'][timetick] = row['sram occupancy']
            # sa util 계산
            start_sa_util = int(start_session + row['current cycle'] - row['compute cycle'])
            end_sa_util = int(start_session + row['current cycle'])
            for timetick in range(max(start_sa_util, 1), min(end_sa_util, total_cycle_max)):
                result['sa util'][timetick] = row['sa util']
            # va util 계산
            for timetick in range(max(start_sa_util, 1), min(end_sa_util, total_cycle_max)):
                result['va util'][timetick] = row['va util']
#            if op_name == 'flash attention':
#                print("FA,", fa_flag,"mem bw - ", start_session, end_mem_bw)
#                print("FA,", fa_flag,"compute - ", start_sa_util, end_sa_util)
        start_session = int(row['total cycle'])

    # Save the result as a CSV file
#    result_df = pd.DataFrame(result)
#    csv_file_name = f"{op_name.replace(' ', '_')}_result.csv"
#    result_df.to_csv(csv_file_name, index=False)
#    print(f"Saved result as {csv_file_name}")
    if save_png:
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(result['timetick'], result['memory bw'], label='Memory BW')
        plt.plot(result['timetick'], result['sram occupancy'], label='SRAM Occupancy')
        plt.plot(result['timetick'], result['sa util'], label='Systolic-array Util')
        plt.plot(result['timetick'], result['va util'], linestyle='--', label='Vector-unit Util')

        plt.xlabel('Timetick')
        plt.ylabel('Values')
        #plt.title(f'Performance Metrics for {op_name}')
        #plt.legend()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)  # 가로로 나열, 그래프 위로 위치
    
        plt.grid(True)
        plt.yticks([0,50,100],[0,50,100])
        plt.xlim(0, 300000) 
#        plt.xlim(0, 16100000) 
        plt.tight_layout()

        # Save the plot as an image
        file_name = f"figure/{file_path[:-4]}/{op_name.replace(' ', '_')}.png"
        directory = f"figure/{file_path[:-4]}/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(file_name)
        plt.close()  # Close the plot to avoid overlap in the next iteration

        print(f"Saved plot as {file_name}")
#print(tot_cycle)
total_sum = sum(tot_cycle)

ratios = [round((value / total_sum) * 100, 2) for value in tot_cycle]

print(tot_cycle)
print(op_order)
print(ratios)
