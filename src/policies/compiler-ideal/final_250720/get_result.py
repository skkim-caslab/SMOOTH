import sys
import csv

def parse_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if len(row) == 6:
                seq_len = int(row[0])
                latency = float(row[1])
                sa_cycle = int(row[2])
                ve_cycle = int(row[3])
                linear_op_cycle = int(row[4])
                non_linear_op_cycle = int(row[5])
                data.append({
                    'seq_len': seq_len,
                    'latency': latency,
                    'sa_cycle': sa_cycle,
                    've_cycle': ve_cycle,
                    'linear_op_cycle': linear_op_cycle,
                    'non_linear_op_cycle': non_linear_op_cycle
                })
    return data

def calculate_metrics(data, last_num):
    # (1) TTFT: Time to First Token (latency of seq_len = 1)
    ttft = next((entry['latency'] for entry in data if entry['seq_len'] == 1), None)
    if ttft is None:
        raise ValueError("No entry found for seq_len = 1")

    # (2) TTLT: Total time from prompt to seq_len = last_num
    final_entry = next((entry for entry in data if entry['seq_len'] == last_num), None)
    if final_entry is None:
        raise ValueError(f"No entry found for seq_len = {last_num}")
    ttlt = sum(entry['latency'] for entry in data if 1 <= entry['seq_len'] <= last_num)

    # (3) Throughput: (prompt + sum of latencies for seq_len 1 to N) / N
    throughputs = []
    cumulative_time = 0
    for n in range(1, last_num + 1):
        entry = next((e for e in data if e['seq_len'] == n), None)
        if entry:
            cumulative_time += entry['latency']
            throughput = n / cumulative_time
            throughputs.append(throughput)
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

    # (4-1) SA util average: SA cycle / linear operation cycle
    # (4-2) SA util average: SA cycle / (linear + non-linear operation cycle)
    sa_util_1 = []
    sa_util_2 = []

    # (5-1) VE util average: VE cycle / non-linear operation cycle
    # (5-2) VE util average: VE cycle / (linear + non-linear operation cycle)
    va_util_1 = []
    va_util_2 = []

    # (6) SA+VE utilization: (SA + VE) / (linear + non-linear)
    total_util = []

    for entry in data:
        if 1 <= entry['seq_len'] <= last_num:
            linear = entry['linear_op_cycle']
            non_linear = entry['non_linear_op_cycle']
            sa = entry['sa_cycle']
            ve = entry['ve_cycle']
            total_op = linear + non_linear

            if linear > 0:
                sa_util_1.append(sa / linear)
            if total_op > 0:
                sa_util_2.append(sa / total_op)
                va_util_2.append(ve / total_op)
                total_util.append((sa + ve) / total_op)
            if non_linear > 0:
                va_util_1.append(ve / non_linear)

    avg_sa_util_1 = sum(sa_util_1) / len(sa_util_1) if sa_util_1 else 0
    avg_sa_util_2 = sum(sa_util_2) / len(sa_util_2) if sa_util_2 else 0
    avg_va_util_1 = sum(va_util_1) / len(va_util_1) if va_util_1 else 0
    avg_va_util_2 = sum(va_util_2) / len(va_util_2) if va_util_2 else 0
    avg_total_util = sum(total_util) / len(total_util) if total_util else 0

    return {
        'TTFT': ttft,
        'TTLT': ttlt,
        'Average Throughput': avg_throughput,
        'Average SA Utilization (SA cycle / linear op cycle)': avg_sa_util_1,
        'Average SA Utilization (SA cycle / (linear + non-linear op cycle))': avg_sa_util_2,
        'Average VA Utilization (VE cycle / non-linear op cycle)': avg_va_util_1,
        'Average VA Utilization (VE cycle / (linear + non-linear op cycle))': avg_va_util_2,
        'Average Total Utilization ((SA+VE) / (linear + non-linear op cycle))': avg_total_util
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename> <last_num>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        last_num = int(sys.argv[2])
        if last_num < 1:
            raise ValueError("last_num must be a positive integer")
        data = parse_file(filename)
        metrics = calculate_metrics(data, last_num)
        # Output only the values, comma-separated
        values = [metrics[key] for key in [
            'TTFT',
            'TTLT',
            'Average Throughput',
            'Average SA Utilization (SA cycle / linear op cycle)',
    #        'Average SA Utilization (SA cycle / (linear + non-linear op cycle))',
    #        'Average VA Utilization (VE cycle / non-linear op cycle)',
    #        'Average VA Utilization (VE cycle / (linear + non-linear op cycle))',
            'Average Total Utilization ((SA+VE) / (linear + non-linear op cycle))'
        ]]
        print(','.join(f"{value:.6f}" for value in values))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
