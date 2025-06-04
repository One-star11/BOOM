import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault
import numpy as np
vault = Vault(rel_dir="vaults", vault_name="5m_vs_6m.vlt", vault_uid="Good")

experience = vault.read().experience

numpy_experience = jax.tree_util.tree_map(lambda x: np.array(x), experience)

print(numpy_experience.keys())
for key in numpy_experience.keys():
    print(key)
    if isinstance(numpy_experience[key], np.ndarray):
        print(numpy_experience[key].shape)
        #print(numpy_experience[key])
        print("--------------------------------")
    elif isinstance(numpy_experience[key], dict):
        for k in numpy_experience[key].keys():
            print(k)
            print(numpy_experience[key][k].shape)
            print("--------------------------------")
            print("--------------------------------")
    else:
        print(type(numpy_experience[key]))
        print("--------------------------------")