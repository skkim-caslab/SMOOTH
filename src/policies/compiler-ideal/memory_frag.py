import sys
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class OperationInfo:
    name: str
    size: int
    reuse: int

class MemoryAllocator:
    SRAM_SIZE = 8*1024 * 1024  # 512KB
    OP_NAMES = [
        'MHA', 'q_projection', 'k_projection', 'v_projection', 'q_mul_k',
        'softmax', 'a_mul_v', 'w0_projection', 'FFN', 'w1_projection', 'gelu', 'w2_projection'
    ]
    NON_LINEAR_OPS = {"MHA", "FFN", "gelu"}
#    PRELOAD_OPS = {"softmax": 0, "MHA": 0, "FFN": 0, "gelu": 0}  # Preload operation tile counts
    PRELOAD_OPS = {"softmax": 10, "MHA": 10, "FFN": 10, "gelu": 10}  # Preload operation tile counts
#    PRELOAD_OPS = {"softmax": 3, "MHA": 3, "FFN": 3, "gelu": 3}  # Preload operation tile counts

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.sram: List[List[int]] = []
        self.time_tick: int = 0
        self.operations: List[Tuple[str, Dict[Tuple[str, ...], List[OperationInfo]]]] = []
        self.reuse_counts: Dict[str, int] = {}
        self.current_op_index: int = 0
        self.current_op_tiles: Dict[Tuple[str, ...], List[OperationInfo]] = {}
        self.fragmentation_stats: List[Dict] = []  # Store fragmentation analysis results
        self.preloaded_tiles: set = set()  # Track preloaded tile names
        self.WIDTH = 1024  # 시각화 가로 크기
        self.HEIGHT = 512  # 시각화 세로 크기

    def _find_free_block(self, sram_row: List[int], size: int) -> Tuple[int, int]:
        """Find first contiguous block of free bits in SRAM row."""
        if not (0 < size <= self.SRAM_SIZE):
            raise ValueError(f"Invalid size: {size}")
        if len(sram_row) != self.SRAM_SIZE:
            raise ValueError("Invalid SRAM row size")

        for start in range(self.SRAM_SIZE - size + 1):
            if all(sram_row[i] == 0 for i in range(start, start + size)):
                return start, start + size - 1
        raise ValueError(f"No contiguous block of size {size} found")

    def _find_holes(self, sram_row: List[int]) -> List[Tuple[int, int]]:
        """Find all contiguous free blocks (holes) in SRAM row."""
        holes = []
        start = None
        for i in range(self.SRAM_SIZE):
            if sram_row[i] == 0 and start is None:
                start = i
            elif sram_row[i] == 1 and start is not None:
                holes.append((start, i - 1))
                start = None
        if start is not None:
            holes.append((start, self.SRAM_SIZE - 1))
        return holes

    def _load_data(self) -> List[OperationInfo]:
        """Load and parse JSON data from file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
                return [OperationInfo(*item) for item in raw_data]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading JSON file: {e}")

    def _get_tile_number(self, op_name: str) -> Tuple[str, ...]:
        """Extract tile number from operation name."""
        return tuple(op_name.split("_")[-4:-1]) if "_" in op_name else ()

    def _is_non_linear_op(self, op_name: str) -> bool:
        """Check if operation is non-linear by comparing base name with NON_LINEAR_OPS."""
        base_name = op_name.split("_")[0] if "_" in op_name else op_name
        return base_name in self.NON_LINEAR_OPS

    def process_operations(self) -> None:
        """Process and organize operations by type and tile."""
        data = self._load_data()
        current_op = self.OP_NAMES[self.current_op_index] if self.OP_NAMES else ""

        for op in data:
            tile_num = self._get_tile_number(op.name)
            op_base_name = op.name.split("_")[0] if "_" in op.name else op.name

            if current_op in op.name:
                if op.name not in self.reuse_counts:
                    self.current_op_tiles.setdefault(tile_num, []).append(op)
            else:
                if self.current_op_tiles:
                    self.operations.append((current_op, self.current_op_tiles))
                    self.current_op_tiles = {}

                self.current_op_index = min(self.current_op_index + 1, len(self.OP_NAMES) - 1) if self.OP_NAMES else 0
                current_op = self.OP_NAMES[self.current_op_index] if self.OP_NAMES else op_base_name

                if op.name not in self.reuse_counts:
                    self.current_op_tiles.setdefault(tile_num, []).append(op)

            if op.reuse > 1:
                self.reuse_counts[op.name] = op.reuse
            elif op.name in self.reuse_counts:
                self.reuse_counts[op.name] -= 1
                if self.reuse_counts[op.name] <= 1:
                    del self.reuse_counts[op.name]

        if self.current_op_tiles:
            self.operations.append((current_op, self.current_op_tiles))

    def _allocate_block(self, time_index: int, start: int, end: int, duration: int) -> None:
        """Mark SRAM block as allocated for specified duration."""
        for i in range(duration):
            if time_index + i < len(self.sram):
                for bit in range(start, end + 1):
                    self.sram[time_index + i][bit] = 1

    def _has_alloc_in_next_tile(self, current_op_index: int, current_tile_index: int, tiles: Dict) -> bool:
        """Check if the next tile has an alloc operation."""
        tile_keys = list(tiles.keys())
        if current_tile_index + 1 < len(tile_keys):
            next_tile_key = tile_keys[current_tile_index + 1]
            next_tile_ops = tiles[next_tile_key]
            return any("alloc" in op.name for op in next_tile_ops)
        return False

    def _get_duration_for_reuse(self, op_index: int, tile_index: int, reuse: int) -> int:
        """Calculate duration for operations with reuse > 1, spanning current and next reuse-1 operations."""
        total_duration = 0
        current_op_tiles = list(self.operations[op_index][1].keys())
        
        total_duration += len(current_op_tiles) - tile_index
        
        for i in range(1, reuse):
            if op_index + i < len(self.operations):
                total_duration += len(self.operations[op_index + i][1])
        
        return total_duration

    def _preload_tiles(self, current_op_index: int, current_tile_index: int, op_name: str, time_tick: int) -> None:
        """Preload tiles for operations in PRELOAD_OPS, considering SRAM status with previous preloads at the same time_tick."""
        base_op_name = op_name.split("_")[0] if "_" in op_name else op_name
        if base_op_name not in self.PRELOAD_OPS:
            return

        preload_count = self.PRELOAD_OPS[base_op_name]
        print(f"Preloading up to {preload_count} tiles for {base_op_name} at time {time_tick}")

        tiles_preloaded = 0
        ops_to_preload = []

        # Collect operations from future tiles, excluding already preloaded tiles
        for op_idx in range(current_op_index, len(self.operations)):
            op, tiles = self.operations[op_idx]
            tile_keys = list(tiles.keys())
            start_tile = current_tile_index + 1 if op_idx == current_op_index else 0

            for tile_idx in range(start_tile, len(tile_keys)):
                if tiles_preloaded >= preload_count:
                    break
                tile_key = tile_keys[tile_idx]
                required_time_tick = base_time_tick + (tile_idx - start_tile) + 1
                for op in tiles[tile_key]:
                    if op.name not in self.preloaded_tiles:
                        ops_to_preload.append(op)
                        tiles_preloaded += 1
            base_time_tick += len(tile_keys) - start_tile
            if tiles_preloaded >= preload_count:
                break

        # Initialize current SRAM state for the given time_tick
        current_sram = self.sram[time_tick - 1].copy() if time_tick > 0 else [0] * self.SRAM_SIZE

        # Attempt to allocate preloaded operations at the same time_tick
        for op in ops_to_preload:
            if "load" in op.name or "alloc" in op.name:
                try:
                    # Extend SRAM if necessary
                    while time_tick >= len(self.sram):
                        self.sram.append([0] * self.SRAM_SIZE)
                    
                    # Find free block in current SRAM state (including previous preloads)
                    start, end = self._find_free_block(current_sram, op.size)

                    # Calculate duration to keep the tile until its required time tick
                    duration = max(1, required_time_tick - time_tick)
                    # If reuse > 1, ensure the duration covers the reuse period
                    if op.reuse > 1:
                        reuse_duration = self._get_duration_for_reuse(current_op_index, current_tile_index, op.reuse)
                        duration += reuse_duration
                    
                    # Update SRAM for the duration of the operation
                    #duration = self._get_duration_for_reuse(current_op_index, current_tile_index, op.reuse) if op.reuse > 1 else op.reuse
                    self._allocate_block(time_tick, start, end, duration)
                    
                    # Update temporary SRAM state for next preload
                    for i in range(start, end + 1):
                        current_sram[i] = 1
                        
                    self.preloaded_tiles.add(op.name)  # Mark tile as preloaded
                    print(f"Preloaded {op.name} at time {time_tick} (based on updated SRAM): {start}-{end} (duration: {duration})")
                except ValueError as e:
                    print(f"Fragmentation error: Cannot preload {op.name} at time {time_tick} due to {str(e)}. Stopping preload.")
                    break

    def analyze_fragmentation(self) -> None:
        """Analyze memory holes and fragmentation for each time step."""
        self.fragmentation_stats = []
        
        for time, sram_row in enumerate(self.sram):
            holes = self._find_holes(sram_row)
            hole_sizes = [(end - start + 1) for start, end in holes]
            total_hole_size = sum(hole_sizes)
            num_holes = len(holes)
            max_hole_size = max(hole_sizes) if hole_sizes else 0
            avg_hole_size = total_hole_size / num_holes if num_holes > 0 else 0
            
            fragmentation_index = 1 - (max_hole_size / self.SRAM_SIZE) if max_hole_size > 0 else 1.0
            
            self.fragmentation_stats.append({
                "time": time,
                "num_holes": num_holes,
                "total_hole_size": total_hole_size,
                "max_hole_size": max_hole_size,
                "avg_hole_size": avg_hole_size,
                "fragmentation_index": fragmentation_index,
                "used_memory": self.SRAM_SIZE - total_hole_size
            })

    def save_fragmentation_stats(self, output_path: str) -> None:
        """Save fragmentation analysis results to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.fragmentation_stats, f, indent=2)
            print(f"Fragmentation stats saved to {output_path}")
        except Exception as e:
            print(f"Error saving fragmentation stats: {e}")

    def print_summary_stats(self) -> None:
        """Print summary statistics for fragmentation analysis."""
        if not self.fragmentation_stats:
            print("No fragmentation stats available")
            return

        avg_num_holes = sum(stat["num_holes"] for stat in self.fragmentation_stats) / len(self.fragmentation_stats)
        avg_total_hole_size = sum(stat["total_hole_size"] for stat in self.fragmentation_stats) / len(self.fragmentation_stats)
        max_max_hole_size = max(stat["max_hole_size"] for stat in self.fragmentation_stats)
        avg_fragmentation_index = sum(stat["fragmentation_index"] for stat in self.fragmentation_stats) / len(self.fragmentation_stats)
        peak_used_memory = max(stat["used_memory"] for stat in self.fragmentation_stats)

        print("\nFragmentation Summary:")
        print(f"Average number of holes: {avg_num_holes:.2f}")
        print(f"Average total hole size: {avg_total_hole_size:.2f} bytes")
        print(f"Maximum hole size: {max_max_hole_size} bytes")
        print(f"Average fragmentation index: {avg_fragmentation_index:.4f}")
        print(f"Peak used memory: {peak_used_memory} bytes ({peak_used_memory / self.SRAM_SIZE * 100:.2f}% of SRAM)")

    def visualize_sram_at_time(self, time_tick: int, save_path: Optional[str] = None) -> None:
        """Visualize SRAM state at given time_tick using Matplotlib (1024x128) and optionally save to file."""
        if not (0 <= time_tick < len(self.sram)):
            print(f"Invalid time_tick: {time_tick}. Must be between 0 and {len(self.sram) - 1}")
            return

        # SRAM 데이터를 1024x128 배열로 변환 및 정규화
        #sram_row = np.array(self.sram[time_tick][:128*1024])
        #sram_row = np.array(self.sram[time_tick][:131072])
        sram_row = np.array(self.sram[time_tick])
        print("50-100",sram_row[50:100])
        print("3900-4100",sram_row[3900:4100])

        
        # -1 값 확인 및 로깅
        if np.any(sram_row < 0):
            print(f"Warning: Negative values (-1) found in SRAM at time {time_tick}. Converting -1 to 0.")
            sram_row = np.where(sram_row < 0, 0, sram_row)
        
        # 0과 1 이외의 값 확인
        if np.any((sram_row != 0) & (sram_row != 1)):
            print(f"Warning: Invalid values found in SRAM at time {time_tick}. Clipping to [0, 1].")
            sram_row = np.clip(sram_row, 0, 1)

        sram_image = sram_row.reshape(self.HEIGHT, self.WIDTH)

        # Matplotlib으로 시각화 (0: 흰색, 1: 검정색)
        plt.figure(figsize=(10, 2))  # 가로 10인치, 세로 2인치
        plt.imshow(sram_image, cmap='binary', interpolation='nearest')
        plt.title(f"SRAM State at Time {time_tick}")
        plt.xlabel("Width (1024 bits)")
        plt.ylabel("Height (512 bits)")
        plt.colorbar(label="Bit Value (0: White, 1: Black)")
        plt.tight_layout()

        # 이미지 저장
        if save_path is None:
            save_path = f"sram_time_{time_tick}.png"
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SRAM visualization saved to {save_path}")
        except Exception as e:
            print(f"Error saving SRAM visualization: {e}")
        finally:
            plt.close()


    def allocate_memory(self) -> List[List[int]]:
        """Allocate memory for operations and return SRAM state."""
        total_time = sum(len(tiles) for _, tiles in self.operations)
        self.sram = [[0] * self.SRAM_SIZE for _ in range(total_time)]
        self.time_tick = 0

        for op_index, (op_name, tiles) in enumerate(self.operations):
            print(f"Operation: {op_name}, Tiles: {len(tiles)}")
            is_non_linear = self._is_non_linear_op(op_name)
            
            if is_non_linear:
                for tile_index, (tile_num, op_list) in enumerate(tiles.items()):
                    for op in op_list:
                        if ("load" in op.name or "alloc" in op.name) and op.name not in self.preloaded_tiles:
                            start, end = self._find_free_block(self.sram[self.time_tick], op.size)
                            duration = self._get_duration_for_reuse(op_index, tile_index, op.reuse) if op.reuse > 1 else op.reuse
                            self._allocate_block(self.time_tick, start, end, duration)
                            print(f"Allocated non-linear {op.name} at time {self.time_tick + 1}: {start}-{end} (duration: {duration})")
                    # Preload after processing all tiles for non-linear op
                    if tile_index == len(tiles) - 1:
                        self._preload_tiles(op_index, tile_index, op_name, self.time_tick + 1)
                self.time_tick += 1
            else:
                for tile_index, (tile_num, op_list) in enumerate(tiles.items()):
                    self.time_tick += 1
                    for op in op_list:
                        if "load" in op.name and op.name not in self.preloaded_tiles:
                            start, end = self._find_free_block(self.sram[self.time_tick - 1], op.size)
                            duration = self._get_duration_for_reuse(op_index, tile_index, op.reuse) if op.reuse > 1 else op.reuse
                            self._allocate_block(self.time_tick - 1, start, end, duration)
                            print(f"Allocated {op.name} at time {self.time_tick}: {start}-{end}")
                        elif "alloc" in op.name and op.name not in self.preloaded_tiles:
                            start, end = self._find_free_block(self.sram[self.time_tick - 1], op.size)
                            has_next_alloc = self._has_alloc_in_next_tile(op_index, tile_index, tiles)
                            duration = (len(tiles) - tile_index if not has_next_alloc and self.time_tick < len(self.sram) else 1)
                            self._allocate_block(self.time_tick - 1, start, end, duration)
                            print(f"Allocated {op.name} at time {self.time_tick}: {start}-{end}")
                    self._preload_tiles(op_index, tile_index, op_name, self.time_tick)

        self.analyze_fragmentation()
        return self.sram

def main() -> None:
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Error: JSON file path required")
        sys.exit(1)

    allocator = MemoryAllocator(sys.argv[1])
    allocator.process_operations()
    sram_state = allocator.allocate_memory()
    print(f"Memory allocation completed. SRAM state shape: {len(sram_state)}x{len(sram_state[0])}")
    
    allocator.visualize_sram_at_time(int(sys.argv[2]))
    
    allocator.save_fragmentation_stats("fragmentation_analysis.json")
    allocator.print_summary_stats()

if __name__ == "__main__":
    main()
