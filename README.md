# BOOM : Bridging Offline Data to Online Learning in Multi-Agent Reinforcement Learning with Self-Supervised Pretraining

<a href="#citation
"><img src="./assets/Figure_DS.png"></a>



## 🎯 Description 
This is a code repository for The Final Project in SNU RL for Data Science.


## ⚙️ Installation
The code is based on pymarl2. 
For detailed information, refere to the installation instructions of [pymarl2](https://github.com/hijkzzz/pymarl2) and [SMAC](https://github.com/oxwhirl/smac).

1️⃣ Cloning BOOM

`git clone https://github.com/One-star11/BOOM.git`

2️⃣ Donwload and setup StarCraftII 

`bash install_sc2.sh`

3️⃣ Install required packages 

`pip install -r requirements.txt`

```
wget https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/5m_vs_6m.zip --show-progress

unzip 3m.zip -d vaults

pip install flashbax~=0.1.2
````

running the code requires about ~40GB of RAM, and ~8GB of VRAM


## 🎮 Running Script

🏃Run an experiment 

`bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)`

✔️ Example 

`bash run.sh qmix sc2 5m_vs_6m use_MT=True 1 0 1`
  
* `use_MT` means executing the model plugs in MA2E into the baseline algorithm. 
