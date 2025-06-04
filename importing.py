import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_episode_lengths(vault, name):
    print(f"\nAnalyzing {name} vault episode lengths:")
    experience = vault.read().experience
    
    # Find episode boundaries
    terminals = np.array(experience["terminals"])
    truncations = np.array(experience["truncations"])
    episode_ends = (terminals[0] > 0.5) | (truncations[0] > 0.5)
    end_indices = np.where(episode_ends)[0]
    
    # Calculate episode lengths
    episode_lengths = np.diff(np.concatenate([[0], end_indices]))
    
    print(f"Number of episodes: {len(episode_lengths)}")
    print(f"Mean length: {np.mean(episode_lengths):.2f}")
    print(f"Std length: {np.std(episode_lengths):.2f}")
    print(f"Min length: {np.min(episode_lengths)}")
    print(f"Max length: {np.max(episode_lengths)}")
    print(f"Length distribution:")
    for length in range(10, 71, 10):
        count = np.sum((episode_lengths >= length-10) & (episode_lengths < length))
        print(f"  {length-10}-{length} steps: {count} episodes ({count/len(episode_lengths)*100:.1f}%)")
    
    return episode_lengths

# Analyze each vault
vault_good = Vault(rel_dir="vaults", vault_name="5m_vs_6m.vlt", vault_uid="Good")
vault_medium = Vault(rel_dir="vaults", vault_name="5m_vs_6m.vlt", vault_uid="Medium")
vault_poor = Vault(rel_dir="vaults", vault_name="5m_vs_6m.vlt", vault_uid="Poor")

good_lengths = analyze_episode_lengths(vault_good, "Good")
medium_lengths = analyze_episode_lengths(vault_medium, "Medium")
poor_lengths = analyze_episode_lengths(vault_poor, "Poor")

# Plot distribution
plt.figure(figsize=(12, 6))
plt.hist([good_lengths, medium_lengths, poor_lengths], 
         label=['Good', 'Medium', 'Poor'], bins=range(0, 71, 5), alpha=0.7)
plt.xlabel('Episode Length')
plt.ylabel('Number of Episodes')
plt.title('Distribution of Episode Lengths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('episode_lengths_distribution.png')
print("\nPlot saved as 'episode_lengths_distribution.png'")

#Good 996727
#Medium 996856
#Poor 934505