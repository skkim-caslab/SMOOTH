import json
import pandas as pd
import glob

# Directory path where JSON files are saved (e.g. "./json_data/*.json")
json_files = glob.glob("./tile_size/*.json")

# List to store all data
data_list = []

# Read the JSON file and add it to the data list
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for op, params in json_data.items():
            entry = {"operation": op}
            for key, value in params.items():
                if key != "next_ops_name": # Excluding 'next_ops_name'
                    entry[key] = value
            data_list.append(entry)

# Create DataFrame
df = pd.DataFrame(data_list)

# Combine duplicate values ​​for the same column of the same operation with a comma
df_grouped = df.groupby("operation").agg(lambda x: ', '.join(map(str, sorted(set(x)))))

# Save as CSV file (can be changed to desired file name)
df_grouped.to_csv("output.csv", encoding="utf-8")

# Print table
print(df_grouped)
