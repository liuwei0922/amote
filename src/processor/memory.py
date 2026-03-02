import numpy as np
import pickle
from typing import List, Optional
from src.core.signal import InternalSignal
from src.processor.base import Processor


class MemoryProcessor(Processor):
    def __init__(self, db_path: str = "brain_memory.pkl"):
        super().__init__()
        self.db_path = db_path
        self.memory_store: dict[str, np.ndarray] = self._load_db()

    def _load_db(self):
        try:
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.memory_store, f)

    def process(self, action: str, signal: InternalSignal) -> None:
        if action == "store":
            self.memory_store[signal.debug_info] = signal.vector
            print(f"  [Memory] 已巩固记忆: {signal.debug_info}")
            self.save_db()

        elif action == "query":
            best_match_info = None
            highest_sim = -1.0

            for info, mem_vec in self.memory_store.items():
                sim = np.dot(signal.vector, mem_vec)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_info = info

            if highest_sim > 0.85 and best_match_info is not None:
                print(
                    f"  [Memory] 联想唤醒: 找到相似记忆 -> {best_match_info} (相似度 {highest_sim:.2f})"
                )
                recalled_signal = InternalSignal(
                    self.memory_store[best_match_info], best_match_info
                )
                self.output_buffer.append(recalled_signal)
            else:
                print(f"  [Memory] 检索失败，没有相关记忆。")
