from torch import Tensor


class InternalSignal:

    def __init__(self, vector: Tensor, debug_info: str = ""):
        self.vector = vector
        self.debug_info = debug_info

    def __repr__(self):
        if self.vector is not None:
            return f"<Signal {self.debug_info} | Dim:{self.vector.shape[0]} | Avg:{self.vector.mean():.3f}>"
        else:
            return f"<Signal {self.debug_info} | No Vector>"
