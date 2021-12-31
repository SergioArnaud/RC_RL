# RC_RL
The purpose of this repo is to train DeepRL models in the VGDL environment. The agents
available are:

* DQN ([Mnih et al., 2015][https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf])
* C51 ([Bellemare et al., 2017][http://proceedings.mlr.press/v70/bellemare17a.html])
* Rainbow ([Hessel et al., 2018][https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680])
* IQN ([Dabney et al., 2018][https://arxiv.org/abs/1806.06923])
* SAC ([Haarnoja et al., 2018][https://arxiv.org/abs/1812.05905])
* EfficientZero ([Ye et al., 2019][https://arxiv.org/abs/2111.00210])

## Requirements

python 3.6 or later (3.8 preferred)
> Note python-dev is needed to run efficientZero

## Setup

```
# First clone the repo
git clone https://github.com/SergioArnaud/vgdl-emulator-py3.git

# Add the submodules
git submodule update --init --recursive

# Install the requirements
pip install -r requirements.txt

# Make the c dependencies for efficientZero (optional - only needed if you want to train an efficientZero agent)
# This might be complicated, specially if your trying to run this on a mac. Linux is recommended.
# Finally, note that python-dev is needed to build the c dependencies. 

cd EfficientZero/config/core/ctree
python setup.py build_ext --inplace

```

You can also use the [docker image](docker/) to train an agent. 

## Usage

#### Dopamine

```
python -um dopamine.discrete_domains.train  \
 --base_dir=./tmp/dopamine/rainbow_ef \ # Where to store the results
 --gin_files='dopamine/jax/agents/full_rainbow/configs/full_efficient_rainbow.gin' \ # The gin file to use
 --gin_bindings='create_atari_environment.game_name="VGDL_aliens"' # The vgdl (or atari) game to play
```

#### EfficientZero

```
python -um main.py 
    --env VGDL_aliens \ # Or an atari env, (BreakoutNoFrameskip-v4 for example)
    --case vgdl \ # Or `atari`
    --opr train \
    --amp_type torch_amp \
    --num_gpus 1 \
    --num_cpus 4 \
    --cpu_actor 1 \
    --gpu_actor 1 \
    --force
```

#### Pytorch Implemenation of DDQN

```
python runDDQN.py -game_name aliens
```


## Credit

```
@misc{tsividis2021humanlevel,
      title={Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning}, 
      author={Pedro A. Tsividis and Joao Loula and Jake Burga and Nathan Foss and Andres Campero and Thomas Pouncy and Samuel J. Gershman and Joshua B. Tenenbaum},
      year={2021},
      eprint={2107.12544},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}



