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
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", args=None, env_info=None):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.heads = 1
        self.args = args  # Store args for agent ID encoding
        self.env_info = env_info  # Store env_info
        self.n_agents = groups["agents"]  # Number of agents
        self.n_actions = scheme["avail_actions"]["vshape"][0]  # Number of possible actions
        #TODO : load a pretrained transformer to search relevant episodes from the offline data
        ###input_shape = full state + action, needs to be changed when using some more.
        max_ep_len = 71

        
        # Load data from all three vaults
        self.vlt_good = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Good") #996727
        self.vlt_medium = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Medium") #996856
        self.vlt_poor = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Poor") #934505
        
        # Process vault data into episodes
        self.offline_episodes = []  # List to store processed episodes
        
        def process_vault_data(vault, vault_type):
            print(f"Processing {vault_type} vault...")
            experience = vault.read().experience
            
            # Convert JAX arrays to numpy
            terminals = np.array(experience["terminals"][0])
            truncations = np.array(experience["truncations"][0])
            states = np.array(experience["infos"]["state"][0])
            actions = np.array(experience["actions"][0])
            rewards = np.array(experience["rewards"][0])
            obs = np.array(experience["observations"][0])
            legals = np.array(experience["infos"]["legals"][0])
            
            # Find episode boundaries
            episode_ends = (terminals > 0.5) | (truncations > 0.5)
            end_indices = np.where(episode_ends)[0]
            
            # Process each episode
            episode_start = 0
            
            for ep_idx, episode_end in enumerate(end_indices):
                episode_length = episode_end - episode_start + 1
                
                # Initialize obs_all and action_all for this episode with max_ep_len
                # Shape: [max_ep_len, n_agents, MT_traj_length, dim]
                obs_all = np.zeros((max_ep_len, self.n_agents, self.args.MT_traj_length, self.args.input_shape), dtype=np.float32)
                action_all = np.zeros((max_ep_len, self.n_agents, self.args.MT_traj_length, 1), dtype=np.float32)
                
                # Set agent IDs in obs_all if needed
                if self.args.obs_agent_id:
                    if self.args.obs_last_action:
                        # Agent ID after obs and last action
                        for idx in range(self.n_agents):
                            obs_all[:, idx, :, self.env_info["obs_shape"] + self.env_info["n_actions"] + idx] = 1
                    else:
                        # Agent ID after obs
                        for idx in range(self.n_agents):
                            obs_all[:, idx, :, self.env_info["obs_shape"] + idx] = 1
                
                # Fill in the observations and actions for each timestep
                for t in range(episode_length):
                    # For each timestep, shift the window and add new data
                    if t > 0:
                        obs_all[t, :, :self.args.MT_traj_length-1] = obs_all[t-1, :, 1:]
                        action_all[t, :, :self.args.MT_traj_length-1] = action_all[t-1, :, 1:]
                    
                    # Add new observation
                    obs_all[t, :, -1, :self.env_info["obs_shape"]] = obs[episode_start + t]
                    
                    # Add last action to observation if needed
                    if self.args.obs_last_action and t > 0:
                        for agent_id in range(self.n_agents):
                            obs_all[t, agent_id, -1, self.env_info["obs_shape"] + int(actions[episode_start + t - 1, agent_id])] = 1
                    
                    # Add new action
                    if t < episode_length:
                        action_all[t, :, -1, 0] = actions[episode_start + t].reshape(-1)
                
                # Create episode data structure
                episode = {
                    'state': th.from_numpy(states[episode_start:episode_end+1]),
                    'obs': th.from_numpy(obs[episode_start:episode_end+1]),
                    'actions': th.from_numpy(actions[episode_start:episode_end+1]),
                    'avail_actions': th.from_numpy(legals[episode_start:episode_end+1]),
                    'reward': th.from_numpy(rewards[episode_start:episode_end+1])[:,0:1],
                    'terminated': th.from_numpy(episode_ends[episode_start:episode_end+1])[:,0:1],
                    'filled': th.ones(episode_length, dtype=th.long),
                    'length': episode_length,
                    'vault_type': vault_type,
                    'episode_idx': ep_idx,
                    'probs': th.zeros((episode_length, self.n_agents, self.env_info["n_actions"]), dtype=th.float),
                    'action_all': th.from_numpy(action_all),  # Shape: [max_ep_len, n_agents, MT_traj_length, 1]
                    'obs_all': th.from_numpy(obs_all)  # Shape: [max_ep_len, n_agents, MT_traj_length, input_shape]
                }
                
                # Add to episodes list
                self.offline_episodes.append(episode)
                
                # Update start for next episode
                episode_start = episode_end + 1
            
            print(f"Processed {len(end_indices)} episodes from {vault_type} vault")
        
        # Process all vaults
        process_vault_data(self.vlt_good, "good")
        process_vault_data(self.vlt_medium, "medium")
        process_vault_data(self.vlt_poor, "poor")
        
        # Pad all episodes to max length
        for episode in self.offline_episodes:
            ep_len = episode['length']
            if ep_len < max_ep_len:
                padding_len = max_ep_len - ep_len
                # Pad each field
                episode['state'] = th.cat([episode['state'], 
                    th.zeros(padding_len, *episode['state'].shape[1:], dtype=episode['state'].dtype)], dim=0)
                episode['obs'] = th.cat([episode['obs'], 
                    th.zeros(padding_len, *episode['obs'].shape[1:], dtype=episode['obs'].dtype)], dim=0)
                episode['actions'] = th.cat([episode['actions'], 
                    th.zeros(padding_len, *episode['actions'].shape[1:], dtype=episode['actions'].dtype)], dim=0)
                episode['avail_actions'] = th.cat([episode['avail_actions'], 
                    th.zeros(padding_len, *episode['avail_actions'].shape[1:], dtype=episode['avail_actions'].dtype)], dim=0)
                episode['reward'] = th.cat([episode['reward'], 
                    th.zeros(padding_len, *episode['reward'].shape[1:], dtype=episode['reward'].dtype)], dim=0)
                episode['reward'] = episode['reward'][:,0:1]
                episode['terminated'] = th.cat([episode['terminated'], 
                    th.zeros(padding_len, *episode['terminated'].shape[1:], dtype=th.bool)], dim=0)
                episode['terminated'] = episode['terminated'][:,0:1]
                episode['probs'] = th.cat([episode['probs'],
                    th.zeros(padding_len, *episode['probs'].shape[1:], dtype=th.float)], dim=0)
                # Pad filled with zeros to mark padded timesteps
                episode['filled'] = th.cat([episode['filled'],
                    th.zeros(padding_len, dtype=th.long)], dim=0)
        
        # Prepare input features for transformer
        self.offline_features = []  # List to store episode features
        self.offline_values = []    # List to store episode values (rewards)
        
        for episode in self.offline_episodes:

            #states_actions should be (state....,action....)
            states_actions = th.cat([
                episode['state'],  # Use full padded sequence
                episode['actions']  # Reshape actions to 2D and use full sequence
            ], dim=-1)
            
            # flatten the full padded episode data - make it (max_ep_len * (state_dim + action_dim),)
            states_actions = states_actions.flatten(start_dim=0, end_dim=1)
            self.offline_features.append(states_actions)
            
            # Use full reward sequence as value
            self.offline_values.append(episode['reward'])
        
        # Stack all features and values
        self.offline_features = th.stack(self.offline_features)  # shape: [n_episodes, max_ep_len * (state_dim + action_dim)]
        self.offline_values = th.stack(self.offline_values)      # shape: [n_episodes, max_ep_len]
        
        print(f"Offline features shape: {self.offline_features.shape}")
        print(f"Offline values shape: {self.offline_values.squeeze(-1).shape}")
        
        # Create the transformer with the prepared data
        self.transformer = CrossAttention(
            self.offline_features.shape[1],  # Input dimension is flattened full episode length
            self.heads,
            self.offline_features.shape[1],
            offline_keys=self.offline_features,
            offline_values=self.offline_values.squeeze(-1)  #squeeze the last dimension
        )
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
        """Get attention scores for a full episode."""
        # Interleave states and actions: state1-action1-state2-action2-...
        batch_size = states.shape[0] if len(states.shape) > 2 else 1
        if len(states.shape) == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            
        # Reshape actions if needed (in case they're [batch, time, n_agents, 1])
        if len(actions.shape) == 4:
            actions = actions.squeeze(-1)  # Remove last dimension if it's 1
            
        # Flatten actions across agent dimension
        actions_flat = actions.reshape(batch_size, max_time, -1)
        
        # Create interleaved sequence
        seq_len = max_time
        interleaved = []
        for t in range(seq_len):
            interleaved.append(states[:, t])
            interleaved.append(actions_flat[:, t])
        
        # Stack and flatten
        states_actions = th.cat(interleaved, dim=-1)  # shape: [batch, (state_dim + action_dim) * max_time]
        
        # Move to device
        states_actions = states_actions.to(self.device)
        
        # Get attention scores
        attention_scores = self.transformer(states_actions)  # (batch, 1, 1)
        return attention_scores.squeeze(), 1  # Remove extra dimensions

    def _retrieve_similar_episodes(self, ep_batch, k=3):
        """Retrieve k most similar episodes for each episode in batch."""
        # Create expanded batch
        expanded_batch = self._create_expanded_batch(ep_batch, k)
        
        # Get state and action data from the episode batch
        states = ep_batch["state"]  # shape: (batch, time, state_dim)
        actions = ep_batch["actions"]  # shape: (batch, time, n_agents, 1)
        max_time = states.shape[1]  # Use full padded length
        
        # Process each episode in the batch
        for b in range(ep_batch.batch_size):
            episode_states = states[b]  # (time, state_dim)
            episode_actions = actions[b]  # (time, n_agents, 1)
            
            # Get attention score for the full padded episode
            attention_scores, _ = self._get_episode_attention(
                episode_states, 
                episode_actions, 
                max_time
            )
            
            # Get top-k indices from the offline data
            _, top_k_indices = th.topk(attention_scores, k=k)
            
            # Copy retrieved episodes from vault data to expanded batch
            for i, top_idx in enumerate(top_k_indices):
                expanded_idx = b * (k + 1) + i + 1  # Original episode at b*(k+1), retrieved episodes follow
                episode = self.offline_episodes[top_idx]
                
                print(f"Episode {b}, Retrieved episode {top_idx} from {episode['vault_type']} vault, original length {episode['length']}")
                
                # Copy data to expanded batch
                for key in ep_batch.data.transition_data.keys():
                    print(f"Transition data key: {key}, shape: {expanded_batch['transition_data'][key][expanded_idx].shape}, episode {key} shape: {episode[key].shape}")
                    expanded_batch['transition_data'][key][expanded_idx] = episode[key].view(*expanded_batch['transition_data'][key][expanded_idx].shape)
                
                # Episode data remains the same as we don't have episode-level data
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