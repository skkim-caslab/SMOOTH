from typing import List
from utils import size

class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        self.word_size:int = word_size

data_type_dict = {"int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4), "w4a8": DataType("w4a8", 0.5)}
char_map = {"a": 0, "b":1, "c":2, "d":3}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["int8"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type
    def copy(self) -> "Tensor":
        return Tensor(self.shape.copy(), self.data_type)
        
    def reshape(self, first_indices: str, second_indices: str) -> "Tensor":
        first = 1
        second = 1
        for i in first_indices:
            first *= self.shape[char_map[i]]
        for i in second_indices:
            second *= self.shape[char_map[i]]
        sliced_shape = [first, second]
        return Tensor(sliced_shape, self.data_type)
