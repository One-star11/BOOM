import torch as th
import numpy as np
from types import SimpleNamespace as SN
from .segment_tree import SumSegmentTree, MinSegmentTree
import random
from modules.layer.cross_atten import CrossAttention
import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault
class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():

            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])
    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", args=None):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.heads = 1
        self.args = args  # Store args for agent ID encoding
        self.n_agents = groups["agents"]  # Number of agents
        self.n_actions = scheme["avail_actions"]["vshape"][0]  # Number of possible actions
        #TODO : load a pretrained transformer to search relevant episodes from the offline data
        ###input_shape = full state + action, needs to be changed when using some more.
        self.chunk_size = 10
        self.stride = 5
        self.input_shape = (self.chunk_size * (scheme["state"]["vshape"] + groups["agents"])) ##state + action for 10 timesteps   
        
        # Load data from all three vaults
        self.vlt_good = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Good") #996727
        self.len_vlt_good = 996727
        self.vlt_medium = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Medium") #996856
        self.len_vlt_medium = 996856
        self.vlt_poor = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Poor") #934505
        self.len_vlt_poor = 934505
        
        # Create a mapping structure for chunks
        self.chunk_mapping = []  # List of tuples (vault_type, episode_idx, start_timestep)
        
        # Process each vault's data
        def process_vault_data(vault, vault_type):
            experience = vault.read().experience
            # Convert JAX arrays to torch tensors
            exp_numpy = jax.tree_util.tree_map(lambda x: np.array(x), experience)
            exp_torch = jax.tree_util.tree_map(lambda x: th.from_numpy(x), exp_numpy)
            
            # Get state and action data (batch, time, ...)
            states = exp_torch["infos"]["state"].squeeze(0)  # Remove batch dim
            actions = exp_torch["actions"].squeeze(0)  # Remove batch dim
            rewards = exp_torch["rewards"].squeeze(0)[..., 0:1] + 1e-3  # shape: (time, 1)
            terminals = exp_torch["terminals"].squeeze(0)[..., 0]  # shape: (time,)
            truncations = exp_torch["truncations"].squeeze(0)[..., 0]  # shape: (time,)
            
            # Convert float 0/1 to boolean by thresholding
            episode_ends = ((terminals > 0.5) | (truncations > 0.5))
            
            # Concatenate state and actions for each timestep
            states_actions = th.cat([states, actions], dim=-1)  # shape: (time, state_dim + action_dim)
            
            chunked_keys = []
            chunked_values = []
            
            # Process each episode separately
            episode_start = 0
            total_timesteps = states.shape[0]
            episode_idx = 0
            
            while episode_start < total_timesteps:
                # Find the end of current episode
                episode_end = episode_start
                while episode_end < total_timesteps and not episode_ends[episode_end]:
                    episode_end += 1
                if episode_end < total_timesteps:
                    episode_end += 1  # Include the terminal state
                
                # Process chunks within this episode
                for chunk_start in range(episode_start, episode_end - self.chunk_size + 1, self.stride):
                    chunk_end = chunk_start + self.chunk_size
                    if chunk_end <= episode_end:  # Only create complete chunks
                        # Get chunk of states and actions
                        chunk_sa = states_actions[chunk_start:chunk_end]
                        # Flatten the chunk into a single vector
                        chunk_sa_flat = chunk_sa.reshape(-1)
                        chunked_keys.append(chunk_sa_flat)
                        
                        # Use the last reward in the chunk as the value
                        chunk_value = rewards[chunk_end - 1]
                        chunked_values.append(chunk_value)
                        
                        # Store mapping information
                        self.chunk_mapping.append((vault_type, episode_idx, chunk_start))
                    else:
                        #although it overlaps, make a chunk that ends in the episode
                        chunk_sa = states_actions[episode_end - self.chunk_size:episode_end]
                        chunk_sa_flat = chunk_sa.reshape(-1)
                        chunked_keys.append(chunk_sa_flat)
                        chunk_value = rewards[episode_end - 1]
                        chunked_values.append(chunk_value)
                        
                        # Store mapping information
                        self.chunk_mapping.append((vault_type, episode_idx, episode_end - self.chunk_size))
                
                # Move to next episode
                episode_start = episode_end
                episode_idx += 1
            
            # Stack all chunks
            if chunked_keys:  # Check if we have any chunks
                return (th.stack(chunked_keys), th.stack(chunked_values))
            else:
                print("No valid chunks found")
                return (th.zeros(0, self.input_shape), th.zeros(0, 1))  # Empty tensors if no valid chunks
        
        # Process all vaults and concatenate their data
        good_keys, good_values = process_vault_data(self.vlt_good, "good")
        medium_keys, medium_values = process_vault_data(self.vlt_medium, "medium")
        poor_keys, poor_values = process_vault_data(self.vlt_poor, "poor")
        
        # Concatenate all data
        self.offline_keys = th.cat([good_keys, medium_keys, poor_keys], dim=0)  # All states and actions
        self.offline_values = th.cat([good_values, medium_values, poor_values], dim=0)  # All rewards
        
        print(f"Total chunks created: {self.offline_keys.shape[0]}")
        print(f"Chunk dimension: {self.offline_keys.shape[1]}")
        
        # Create the transformer with the prepared data
        self.transformer = CrossAttention(self.input_shape, self.heads, self.input_shape, 
                                        offline_keys=self.offline_keys,
                                        offline_values=self.offline_values)
        if device != "cpu":
            self.transformer = self.transformer.to(device)
        ###

    def _create_expanded_batch(self, ep_batch, k):
        """Create a new batch with space for original and retrieved episodes.
        
        Args:
            ep_batch: Original episode batch
            k: Number of similar episodes to retrieve per original episode
            
        The expanded batch will have the following structure:
        - For each original episode, we allocate k+1 spaces:
            - 1 for the original episode
            - k for the similar episodes
        - Original episodes are placed at indices: 0, k+1, 2(k+1), 3(k+1), ...
        - Similar episodes will be placed in between
        """
        expanded_batch = {
            'transition_data': {},
            'episode_data': {}
        }
        
        # Initialize expanded batch with same structure as original
        for key in ep_batch.data.transition_data.keys():
            shape = list(ep_batch.data.transition_data[key].shape)
            shape[0] *= (k + 1)  # Multiply batch dimension by k+1
            # Only retrieved episodes should be truncated to chunk_size, original episodes keep full length
            expanded_batch['transition_data'][key] = th.zeros(shape, 
                dtype=ep_batch.data.transition_data[key].dtype,
                device=ep_batch.data.transition_data[key].device)
        
        for key in ep_batch.data.episode_data.keys():
            shape = list(ep_batch.data.episode_data[key].shape)
            shape[0] *= (k + 1)  # Multiply batch dimension by k+1
            expanded_batch['episode_data'][key] = th.zeros(shape,
                dtype=ep_batch.data.episode_data[key].dtype,
                device=ep_batch.data.episode_data[key].device)
        
        # Place original episodes at indices 0, k+1, 2(k+1), ...
        stride = k + 1  # Distance between original episodes in expanded batch
        for key in ep_batch.data.transition_data.keys():
            expanded_batch['transition_data'][key][::stride] = ep_batch.data.transition_data[key]
        
        for key in ep_batch.data.episode_data.keys():
            expanded_batch['episode_data'][key][::stride] = ep_batch.data.episode_data[key]
            
        return expanded_batch

    def _get_episode_attention(self, states, actions, max_time):
        """Create chunks from episode data and get attention scores in one pass."""
        # Concatenate states and actions
        states_actions = th.cat([states, actions], dim=-1)
        chunks = []
        
        # Create chunks that don't cross episode boundaries
        for t in range(0, max_time - self.chunk_size + 1, self.stride):
            chunk = states_actions[t:t + self.chunk_size]
            chunk_flat = chunk.reshape(-1)  # Flatten to match transformer input
            chunks.append(chunk_flat)
        
        # Add final chunk if needed
        if max_time > self.chunk_size and max_time % self.stride != 0:
            final_chunk = states_actions[max_time - self.chunk_size:max_time]
            final_chunk_flat = final_chunk.reshape(-1)
            chunks.append(final_chunk_flat)
        
        if not chunks:
            return None, 0
            
        # Stack all chunks and prepare for transformer
        episode_chunks = th.stack(chunks)  # (n_chunks, chunk_size * (state_dim + n_agents))
        # Add thread dimension for transformer: (n_chunks, 1, feature_dim)
        episode_chunks = episode_chunks.unsqueeze(1)
        
        # Move chunks to same device as transformer
        episode_chunks = episode_chunks.to(self.device)
        
        # Get attention scores in one pass
        attention_scores = self.transformer(episode_chunks)  # (n_chunks, 1, 1)
        return attention_scores.squeeze(), len(chunks)  # Remove extra dimensions

    def _retrieve_similar_episodes(self, ep_batch, k=3):
        """Retrieve k most similar episodes for each episode in batch."""
        # Create expanded batch
        expanded_batch = self._create_expanded_batch(ep_batch, k)
        
        # Get state and action data from the episode batch
        states = ep_batch["state"]  # shape: (batch, time, state_dim)
        actions = ep_batch["actions"]  # shape: (batch, time, n_agents, 1)
        max_time = ep_batch.max_t_filled()
        
        # Process each episode in the batch
        for b in range(ep_batch.batch_size):
            episode_states = states[b]  # (time, state_dim)
            episode_actions = actions[b, :, :, 0]  # (time, n_agents)
            
            attention_scores, n_chunks = self._get_episode_attention(episode_states, episode_actions, max_time)
            if n_chunks == 0:
                continue
                
            # Get top-k indices from the offline data
            _, top_k_indices = th.topk(attention_scores, k=min(k, n_chunks))
            
            # Copy retrieved episodes from vault data to expanded batch
            for i, top_idx in enumerate(top_k_indices):
                expanded_idx = b * (k + 1) + i + 1  # Original episode at b*(k+1), retrieved episodes follow
                chunk_score = attention_scores[top_idx]
                
                # Get the vault, episode index and timestep from our mapping
                vault_type, episode_idx, start_timestep = self.chunk_mapping[top_idx]
                
                # Select the appropriate vault
                if vault_type == "good":
                    vault = self.vlt_good
                elif vault_type == "medium":
                    vault = self.vlt_medium
                else:  # poor
                    vault = self.vlt_poor
                
                print(f"Episode {b}, Retrieved chunk with Score {chunk_score:.4f}, from {vault_type} vault episode {episode_idx}, start {start_timestep}")
                
                # Get data from vault for this episode
                vault_data = vault.read().experience
                
                # Copy data to expanded batch
                for key in ep_batch.data.transition_data.keys():
                    if key == "state":
                        states = th.from_numpy(np.array(vault_data["infos"]["state"][0, episode_idx]))
                        chunk_data = states[start_timestep:start_timestep+self.chunk_size]
                        # Place chunk at the beginning of the episode
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data
                        # Zero-pad the rest of the episode
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "obs":
                        obs = th.from_numpy(np.array(vault_data["observations"][0, episode_idx]))
                        chunk_data = obs[start_timestep:start_timestep+self.chunk_size]
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "actions":
                        actions = th.from_numpy(np.array(vault_data["actions"][0, episode_idx]))
                        chunk_data = actions[start_timestep:start_timestep+self.chunk_size]
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data.unsqueeze(-1)
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "avail_actions":
                        legals = th.from_numpy(np.array(vault_data["infos"]["legals"][0, episode_idx]))
                        chunk_data = legals[start_timestep:start_timestep+self.chunk_size]
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "reward":
                        rewards = th.from_numpy(np.array(vault_data["rewards"][0, episode_idx]))
                        chunk_data = rewards[start_timestep:start_timestep+self.chunk_size]
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "terminated":
                        terminals = th.from_numpy(np.array(vault_data["terminals"][0, episode_idx]) > 0.5)
                        truncations = th.from_numpy(np.array(vault_data["truncations"][0, episode_idx]) > 0.5)
                        term_trunc = terminals | truncations
                        chunk_data = term_trunc[start_timestep:start_timestep+self.chunk_size]
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = chunk_data
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "filled":
                        expanded_batch['transition_data'][key][expanded_idx, :self.chunk_size] = 1
                        if self.chunk_size < expanded_batch['transition_data'][key].shape[1]:
                            expanded_batch['transition_data'][key][expanded_idx, self.chunk_size:] = 0
                    elif key == "action_all":
                        # Match parallel_runner.py format: (batch_size, n_agents, MT_traj_length, 1)
                        actions = th.from_numpy(np.array(vault_data["actions"][0, episode_idx]))
                        # Create sliding window view of actions
                        chunk_data = actions[start_timestep:start_timestep+self.chunk_size].reshape(1, self.n_agents, -1)
                        traj_length = min(self.chunk_size, self.scheme["action_all"]["vshape"][0])
                        # Fill the trajectory with sliding window
                        for t in range(traj_length):
                            expanded_batch['transition_data'][key][expanded_idx, :, t, 0] = chunk_data[0, :, t]
                        # Zero-pad the rest
                        if traj_length < expanded_batch['transition_data'][key].shape[2]:
                            expanded_batch['transition_data'][key][expanded_idx, :, traj_length:, 0] = 0
                    elif key == "obs_all":
                        # Match parallel_runner.py format: (batch_size, n_agents, MT_traj_length, input_shape)
                        obs = th.from_numpy(np.array(vault_data["observations"][0, episode_idx]))
                        # Create sliding window view of observations
                        chunk_data = obs[start_timestep:start_timestep+self.chunk_size].reshape(1, self.n_agents, -1)
                        traj_length = min(self.chunk_size, self.scheme["obs_all"]["vshape"][0])
                        input_shape = chunk_data.shape[-1]
                        # Fill the trajectory with sliding window
                        for t in range(traj_length):
                            expanded_batch['transition_data'][key][expanded_idx, :, t, :input_shape] = chunk_data[0, :, t]
                        # Zero-pad the rest
                        if traj_length < expanded_batch['transition_data'][key].shape[2]:
                            expanded_batch['transition_data'][key][expanded_idx, :, traj_length:, :] = 0
                        
                        # Handle agent ID one-hot encoding if needed (following parallel_runner.py logic)
                        if self.args.obs_agent_id:
                            for idx in range(self.n_agents):
                                if self.args.obs_last_action:
                                    expanded_batch['transition_data'][key][expanded_idx, idx, :traj_length, input_shape + self.n_actions + idx] = 1
                                else:
                                    expanded_batch['transition_data'][key][expanded_idx, idx, :traj_length, input_shape + idx] = 1
                
                # Episode data remains the same as we don't have episode-level data in chunks
                for key in ep_batch.data.episode_data.keys():
                    expanded_batch['episode_data'][key][expanded_idx] = ep_batch.data.episode_data[key][b]
        
        # Create new episode batch with expanded data
        return EpisodeBatch(
            scheme=ep_batch.scheme,
            groups=ep_batch.groups,
            batch_size=ep_batch.batch_size * (k + 1),
            max_seq_length=ep_batch.max_seq_length,
            data=SN(**expanded_batch),
            device=ep_batch.device
        )

    def insert_episode_batch(self, ep_batch, recursive=False):
        """Insert episode batch into buffer with similar episode retrieval."""
        if not recursive:
            # Retrieve similar episodes and expand batch
            ep_batch = self._retrieve_similar_episodes(ep_batch, k=3)
        
        # Continue with original episode insertion logic
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :], recursive=True)
            self.insert_episode_batch(ep_batch[buffer_left:, :], recursive=True)

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            #Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


# Adapted from the OpenAI Baseline implementations (https://github.com/openai/baselines)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, beta, t_max, preprocess=None, device="cpu"):
        super(PrioritizedReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length,
                                                      preprocess=preprocess, device="cpu")
        self.alpha = alpha
        self.beta_original = beta
        self.beta = beta
        self.beta_increment = (1.0 - beta) / t_max
        self.max_priority = 1.0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

    def insert_episode_batch(self, ep_batch):
        # TODO: convert batch/episode to idx?
        pre_idx = self.buffer_index
        super().insert_episode_batch(ep_batch)
        idx = self.buffer_index
        if idx >= pre_idx:
            for i in range(idx - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
        else:
            for i in range(self.buffer_size - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
            for i in range(self.buffer_index):
                self._it_sum[i] = self.max_priority ** self.alpha
                self._it_min[i] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.episodes_in_buffer - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, t):
        assert self.can_sample(batch_size)
        self.beta = self.beta_original + (t * self.beta_increment)

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.episodes_in_buffer) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.episodes_in_buffer) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return self[idxes], idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.episodes_in_buffer
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)