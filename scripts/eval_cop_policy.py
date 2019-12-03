import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
sys.path.append(sys.path[0] + '/..')

from lib.models import get_model_class
from util.datasets import load_dataset
from util.environments import load_environment, generate_rollout

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_classification_loss(rollout_states, rollout_actions):
    pass

# model_loc: location of trained policy file
# length: length of sampled trajectories
# trials: number of rollouts
# config_file: location of config file (I believe it's in configs/fruit_fly_configs)
def compute_stylecon(model_loc, length, trials, config_file):
    state_dict = torch.load(model_loc, map_location=lambda storage, loc: storage)
    with open(config_file) as f:
        config = json.load(f)
    model_config = config['model_config']
    data_config = config['data_config']
    model_class = get_model_class(model_config['name'].lower())
    model = model_class(model_config)
    model.filter_and_load_state_dict(state_dict)
    env = load_environment(data_config['name'])
    losses = []
    for i in range(trials):
        with torch.no_grad():
            env.reset()
            model.reset_policy()
            rollout_states, rollout_actions = generate_rollout(env, model, horizon=length)
            rollout_states = rollout_states.transpose(0,1)
            rollout_actions = rollout_actions.transpose(0,1)
            losses.append(get_classification_loss(rollout_states, rollout_actions))
    print(np.mean(losses))
    return np.mean(losses)


compute_stylecon("saved_experiments/saved_preliminary_copulation_ctvae/fruit_fly_configs/ctvae/best.pth", 100, 3, "saved_experiments/saved_preliminary_copulation_ctvae/fruit_fly_configs/configs/ctvae.json")
