import torch
import torch.nn as nn
import torch.nn.functional as F
from ace_utils import StateActionEncoder
# Importing the MLP and RelationAggregator from ACE

#should be able to handle episode data. 
#episode data is (total_episode_size,timestep, feature_dim). where feature_dim = (state_dim + action_dim)


class CrossAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size, offline_keys=None, offline_values=None,
                 state_len=None, relation_len=None, action_dim=None, use_ace_encoder=True, 
                 agent_num=None, checkpoint_path=None):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size
        self.use_ace_encoder = use_ace_encoder
        
        # Initialize ACE encoder if requested
        if use_ace_encoder and state_len is not None and relation_len is not None:
            if agent_num is None:
                raise ValueError("agent_num is required when using ACE encoder")
                
            self.state_action_encoder = StateActionEncoder(
                agent_num=agent_num,
                state_len=state_len,
                relation_len=relation_len,
                hidden_len=embed_size
            )
            # The StateActionEncoder outputs concatenated [state, action_embed]
            # where state has shape (batch, agent_num, hidden_len) and 
            # action_embed has shape (batch, agent_num, agent_num, hidden_len)
            # After taking mean, we get (batch, hidden_len)
            actual_input_size = embed_size

            # Load the learned ACE encoder weights if checkpoint path is provided
            if checkpoint_path is not None:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Extract state dict (handle different checkpoint formats)
                    if 'state_dict' in checkpoint:
                        full_state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        full_state_dict = checkpoint['model']
                    else:
                        full_state_dict = checkpoint
                    
                    # Define the component keys we want to extract
                    encoder_components = [
                        '_action_encoder',
                        '_state_encoder', 
                        '_relation_encoder',
                        '_relation_aggregator'
                    ]
                    
                    # Filter relevant weights
                    encoder_state_dict = {}
                    for key, value in full_state_dict.items():
                        # Remove potential model prefix if present
                        clean_key = key
                        if clean_key.startswith('model.'):
                            clean_key = clean_key[6:]  # Remove 'model.' prefix
                        
                        # Check if this key belongs to any encoder component
                        for component in encoder_components:
                            if clean_key.startswith(component):
                                encoder_state_dict[clean_key] = value
                                break
                    
                    if encoder_state_dict:
                        # Load the filtered weights
                        missing_keys, unexpected_keys = self.state_action_encoder.load_state_dict(
                            encoder_state_dict, strict=False)
                        
                        print(f"Loaded {len(encoder_state_dict)} parameters for StateActionEncoder")
                        if missing_keys:
                            print(f"Missing keys: {missing_keys}")
                        if unexpected_keys:
                            print(f"Unexpected keys: {unexpected_keys}")
                    else:
                        print("Warning: No matching encoder components found in checkpoint")
                        
                except FileNotFoundError:
                    print(f"Warning: Checkpoint file not found at {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint: {e}")
        
        else:
            self.state_action_encoder = None
            actual_input_size = input_size

        self.toqueries = nn.Linear(actual_input_size, self.emb_size * heads, bias=False)
        
        # offline parameters
        if offline_keys is not None and offline_values is not None:
            # offline_keys: (#episodes, state_action_dim) or raw obs format
            # offline_values: (#episodes, reward_steps)
            if isinstance(offline_keys, dict):
                # Process offline keys through encoder if they're in raw format
                if self.state_action_encoder is not None:
                    with torch.no_grad():
                        processed_keys = []
                        for episode_obs in offline_keys:
                            encoded = self.state_action_encoder(episode_obs)
                            # Take mean across entities if multi-agent
                            if encoded.dim() == 3:
                                encoded = encoded.mean(dim=1)
                            processed_keys.append(encoded)
                        offline_keys = torch.stack(processed_keys)
            
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
        # Handle raw observation input
        if isinstance(x, dict) and self.state_action_encoder is not None:
            # x is raw observation dict
            encoded_output = self.state_action_encoder(x)
            # StateActionEncoder returns concatenated [state, action_embed]
            # Take mean across entities/agents to get final feature vector
            if encoded_output.dim() == 3:
                x = encoded_output.mean(dim=1)  # (batch, hidden_len)
            else:
                x = encoded_output
        
        b, hin = x.size()    # (batch_size, input_size)
        
        # Update input size check to use actual encoded size
        expected_size = self.emb_size if self.state_action_encoder is not None else self.input_size
        assert hin == expected_size, \
                f"Input shape mismatch: expected {expected_size}, got {hin}"

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