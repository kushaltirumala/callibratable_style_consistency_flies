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


def visualize_samples_ctvae(exp_dir, trial_id, num_samples, num_values, repeat_index, burn_in, temperature):
    print('#################### Trial {} ####################'.format(trial_id))

    # Get trial folder
    trial_dir = os.path.join(exp_dir, trial_id)
    assert os.path.isfile(os.path.join(trial_dir, 'summary.json'))

    # Load config
    with open(os.path.join(exp_dir, 'configs', '{}.json'.format(trial_id)), 'r') as f:
        config = json.load(f)
    data_config = config['data_config']
    model_config = config['model_config']

    # Load dataset
    dataset = load_dataset(data_config)
    dataset.eval()
    print(type(dataset))

    # Load best model
    state_dict = torch.load(os.path.join(trial_dir, 'best.pth'), map_location=lambda storage, loc: storage)
    model_class = get_model_class(model_config['name'].lower())
    assert model_class.requires_labels
    model_config['label_functions'] = dataset.active_label_functions
    model = model_class(model_config)
    model.filter_and_load_state_dict(state_dict)

    # Load environment
    env = load_environment(data_config['name'])  # TODO make env_config?


    # TODO for now, assume just one active label function
    # assert len(dataset.active_label_functions) == 1

    # for lf in dataset.active_label_functions:
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    (states, actions, labels_dict) = next(iter(loader))

    if repeat_index >= 0:
        states_single = states[repeat_index].unsqueeze(0)
        states = states_single.repeat(num_samples, 1, 1)

        actions_single = actions[repeat_index].unsqueeze(0)
        actions = actions_single.repeat(num_samples, 1, 1)

    states = states.transpose(0, 1)
    actions = actions.transpose(0, 1)

    y = labels_dict["copulation"]
    with torch.no_grad():
        env.reset(init_state=states[0].clone())
        model.reset_policy(labels=y, temperature=args.temperature)

        rollout_states, rollout_actions = generate_rollout(env, model, burn_in=args.burn_in,
                                                           burn_in_actions=actions, horizon=actions.size(0))
        rollout_states = rollout_states.transpose(0, 1)
        rollout_actions = rollout_actions.transpose(0, 1)


        dataset.save(
            rollout_states,
            rollout_actions,
            labels=y,
            lf_list=dataset.active_label_functions,
            burn_in=burn_in,
            save_path=os.path.join(trial_dir, 'results', "copulating"),
            save_name='repeat_{:03d}_{}'.format(repeat_index, "copulating") if repeat_index >= 0 else '',
            single_plot=(repeat_index >= 0))


if __name__ == '__main__':
    visualize_samples_ctvae("")
