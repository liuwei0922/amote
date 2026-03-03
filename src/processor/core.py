import torch
import torch.nn as nn
from .memory import GraphMemory


class CoreProcessor(nn.Module):
    def __init__(self, core_dim=128):
        super().__init__()
        self.dim = core_dim

        self.memory = GraphMemory(dim=core_dim)

        self.fusion_net = nn.Sequential(
            nn.Linear(core_dim, core_dim), nn.LayerNorm(core_dim), nn.ReLU()  # 激活
        )

        self.op_net = nn.Linear(core_dim, core_dim)

    def forward(self, input_tensor):
        """
        input_tensor: [Batch, SeqLen, Dim]
        """
        batch_size, seq_len, dim = input_tensor.size()

        enhanced_inputs = []

        for b in range(batch_size):
            seq_enhanced = []
            for s in range(seq_len):
                token = input_tensor[b, s, :]

                recalled_list, weights_list = self.memory.query(token)

                if recalled_list:
                    mem_stack = torch.stack(recalled_list)
                    w_stack = torch.tensor(
                        weights_list, device=token.device
                    ).unsqueeze(1)
                    
                    w_norm = torch.softmax(w_stack, dim=0)
                    memory_context = torch.sum(mem_stack * w_norm, dim=0)
                else:
                    memory_context = torch.zeros_like(token)

                fused = token + memory_context
                seq_enhanced.append(fused)

            enhanced_inputs.append(torch.stack(seq_enhanced))

        working_tensor = torch.stack(enhanced_inputs)

        features = self.fusion_net(working_tensor)

        output = self.op_net(features)

        self.last_io_pair = (input_tensor.detach(), output.detach())

        return output

    def update_memory(self, mask=None):
        if hasattr(self, 'last_io_pair'):
            inputs, outputs = self.last_io_pair
            
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            
            if mask is None:
                mask = torch.ones(batch_size, dtype=torch.bool, device=inputs.device)
                
            for b in range(batch_size):
                if mask[b]:
                    for s in range(seq_len):
                        if outputs.dim() == 3:
                            inp = inputs[b, s, :]
                            out = outputs[b, s, :]
                            self.memory.link(inp, out)
                        elif outputs.dim() == 2:
                            inp = inputs[b, s, :]
                            out = outputs[b, :]
                            self.memory.link(inp, out)
