import torch
import torch.nn as nn
import torch.nn.functional as F
#should be able to handle episode data. 
#episode data is (total_episode_size,timestep, feature_dim). where feature_dim = (state_dim + action_dim)

class CrossAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size, offline_keys=None, offline_values=None):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        
        # offline parameters
        if offline_keys is not None and offline_values is not None:
            # offline_keys: (#episodes, state_action_dim)
            # offline_values: (#episodes, reward_steps)
            n_episodes = offline_keys.size(0)
            self.n_episodes = n_episodes
            
            # Pre-expand keys for heads: (heads, n_episodes, embed_size)
            self.register_buffer('keys', 
                offline_keys.unsqueeze(0).repeat(heads, 1, 1)
            )
            # Convert step-wise rewards to episode-wise absolute mean rewards: (n_episodes,)
            self.register_buffer('values',
                offline_values.abs().mean(dim=-1)  # Take absolute mean across time steps
            )
        else:
            raise ValueError("Offline keys and values must be provided")

    def forward(self, x, curiosity_score=None):
        b, hin = x.size()    # (batch_size, input_size)
        assert hin == self.input_size, f'Input size {hin} should match {self.input_size}'
        
        h = self.heads
        e = self.emb_size
        
        # Transform to queries: (b, h, e)
        queries = self.toqueries(x).view(b, h, e)
        queries = queries / (e ** (1/4))

        # Compute attention scores for each batch and head
        # queries: (b, h, e), keys: (h, n_episodes, e)
        # -> attention: (b, h, n_episodes)
        dot = torch.matmul(queries, self.keys.transpose(-2, -1))
        
        # Add episode values to attention scores: (b, h, n_episodes)
        dot = dot + self.values.view(1, 1, -1)
        
        # Apply softmax over episodes dimension: (b, h, n_episodes)
        dot = F.softmax(dot, dim=-1)
        
        if curiosity_score is not None:
            # curiosity_score: (b, score) -> (b, 1, 1)
            curiosity = torch.exp(curiosity_score).view(b, 1, 1)
            dot = dot * curiosity
            dot = F.softmax(dot, dim=-1)  # Renormalize
        
        # Average attention scores across heads: (b, n_episodes)
        out = dot.mean(dim=1)
        
        return out