import re
import statistics
import ast

def find_max_and_avg_consecutive_per_line(filename):
    # Read the file
    with open(filename, 'r') as file:
        content = file.read()

    # Initialize list to store max consecutive lengths for each SRAM status list
    max_lengths = []

    # Split content by 'sram status' with or without space
    blocks = re.split(r'sram status\s*:', content)
    
    for block in blocks:
        block = block.strip()
        if block:  # Ignore empty blocks
            try:
                # Parse the block safely
                sram_status = ast.literal_eval(block)

                # Find the max consecutive length for this SRAM status list
                addresses = [[0, 0]]
                for entry in sram_status:
                    if addresses[-1][1] + 1 == entry[1]:
                        addresses[-1][1] = entry[2]
                    else:
                        addresses.append([entry[1], entry[2]])
                
                # Calculate max length for this block
                line_max_length = max(end - start + 1 for start, end in addresses)
                max_lengths.append(line_max_length)
                
                print("MEM:", addresses)

            except (SyntaxError, ValueError) as e:
                print(f"Error parsing block: {block[:50]}...: {e}")
                continue

    # Calculate max and average of the max consecutive lengths
    if not max_lengths:
        return 0, 0.0

    max_of_max_lengths = max(max_lengths)
    avg_of_max_lengths = statistics.mean(max_lengths)

    return max_of_max_lengths, avg_of_max_lengths

# Example usage
if __name__ == "__main__":
    filename = "tmp"
    try:
        max_len, avg_len = find_max_and_avg_consecutive_per_line(filename)
        print(f"Maximum consecutive address length across all lines: {max_len}")
        print(f"Average of maximum consecutive address lengths per line: {avg_len:.2f}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
