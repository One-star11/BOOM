import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size, offline_keys=None, offline_values=None):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        
        # offline parameters
        if offline_keys is not None and offline_values is not None:
            data_length = offline_keys.size(0)
            self.data_length = data_length
            
            # Pre-expand keys for heads: (heads, data_length, embed_size)
            self.register_buffer('keys', 
                offline_keys.view(1, data_length, embed_size).repeat(heads, 1, 1)
            )
            # Pre-expand values for heads: (heads, data_length, 1)
            self.register_buffer('values',
                offline_values.view(1, data_length, 1).repeat(heads, 1, 1)
            )
        else:
            raise ValueError("Offline keys and values must be provided")

    def forward(self, x, curiosity_score=None):
        b, t, hin = x.size()    # (batch_size, thread_length, input_size)
        assert hin == self.input_size, f'Input size {hin} should match {self.input_size}'
        
        h = self.heads
        e = self.emb_size
        
        # Combine batch and thread: (b*t, h, e)
        queries = self.toqueries(x).view(b*t, h, e)
        queries = queries / (e ** (1/4))

        # Reshape for bmm:
        # queries: (b*t*h, 1, e)
        # keys: (b*t*h, e, data_length)
        queries = queries.view(b*t*h, 1, e)
        keys = self.keys.unsqueeze(0).expand(b*t, -1, -1, -1)  # (b*t, h, data_length, e)
        keys = keys.view(b*t*h, self.data_length, e).transpose(-2, -1)  # (b*t*h, e, data_length)
        
        # Compute attention scores: (b*t*h, 1, data_length)
        dot = torch.bmm(queries, keys)
        
        # Reshape back to (b*t, h, data_length)
        dot = dot.view(b*t, h, self.data_length)
        
        # Apply softmax over data_length dimension: (b*t, h, data_length)
        dot = F.softmax(dot, dim=-1)
        
        # Values are in shape (h, data_length, 1)
        # Expand values to (b*t, h, data_length, 1) - using expand for memory efficiency
        values = self.values.unsqueeze(0).expand(b*t, -1, -1, -1)
        ###TODO - implement curiosity score
        if curiosity_score is not None:
            # Reshape curiosity score: (b*t, 1, 1), since its input is (b,t,1)
            curiosity = curiosity_score.view(b*t, 1, 1)
            # Expand curiosity to match values shape: (b*t, h, data_length, 1)
            curiosity = curiosity.unsqueeze(1).expand(-1, h, self.data_length, -1)
            values = values + curiosity
        
        # Reshape for final bmm:
        # dot: (b*t*h, 1, data_length)
        # values: (b*t*h, data_length, 1)
        dot = dot.view(b*t*h, 1, self.data_length)
        values = values.view(b*t*h, self.data_length, 1)
        
        # Compute attention: (b*t*h, 1, 1)
        out = torch.bmm(dot, values)
        
        # Reshape and average across heads: (b, t, data_length)
        out = out.view(b*t, h, 1).mean(dim=1)
        out = out.view(b, t, 1)
        
        return out