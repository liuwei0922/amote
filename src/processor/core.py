import torch
import torch.nn as nn

class CoreProcessor(nn.Module):
    def __init__(self, core_dim=128, memory_slots=64):
        super().__init__()
        self.dim = core_dim
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, core_dim))
        self.memory_ops = nn.Parameter(torch.randn(memory_slots, core_dim, core_dim))
        
        self.query_proj = nn.Linear(core_dim, core_dim)

    def forward(self, input_tensor):
        """
        Input:  [Batch, SeqLen, Dim]
        Output: [Batch, SeqLen, Dim]  <-- 必须保持序列长度，交给 System 去加权
        """
        batch_size, seq_len, dim = input_tensor.size()
        
        queries = self.query_proj(input_tensor)
        
        attn_logits = torch.matmul(queries, self.memory_keys.t())
        attn_weights = torch.softmax(attn_logits / (dim ** 0.5), dim=-1)
        
        flat_ops = self.memory_ops.view(self.memory_keys.size(0), -1)
        dynamic_ops_flat = torch.matmul(attn_weights, flat_ops)
        dynamic_ops = dynamic_ops_flat.view(batch_size, seq_len, dim, dim)
        input_expanded = input_tensor.unsqueeze(-1)
        processed = torch.matmul(dynamic_ops, input_expanded).squeeze(-1)
        return processed