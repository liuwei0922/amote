from abc import ABC, abstractmethod
from typing import List, Any


class Processor(ABC):

    def __init__(self):
        self.output_buffer: List[Any] = []

    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        pass

    def fetch_signals(self) -> List[Any]:
        signals = self.output_buffer.copy()
        self.output_buffer.clear()
        return signals

    def clear_buffer(self) -> None:
        self.output_buffer.clear()

    @abstractmethod
    def parameters(self) -> List[Any]:
        return []