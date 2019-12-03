import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from keras.models import load_model
import tensorflow as tf

import sys

sys.path.append(sys.path[0] + '/..')

from lib.models import get_model_class
from util.datasets import load_dataset
from util.environments import load_environment, generate_rollout

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_classification_loss(rollout_states, rollout_actions):
    model = load_model('classifier.h5')
    temp = rollout_states.numpy()
    return model.predict(temp)

def visualize_samples_ctvae(exp_dir, trial_id, num_samples, num_values, repeat_index, burn_in, temperature, bad_experiment=True):
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

    # Load best model
    state_dict = torch.load(os.path.join(trial_dir, 'best.pth'), map_location=lambda storage, loc: storage)
    model_class = get_model_class(model_config['name'].lower())
    assert model_class.requires_labels
    model_config['label_functions'] = dataset.active_label_functions
    model = model_class(model_config)
    model.filter_and_load_state_dict(state_dict)

    # Load environment
    env = load_environment(data_config['name'])  # TODO make env_config?

    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    (states, actions, labels_dict) = next(iter(loader))

    if repeat_index >= 0:
        states_single = states[repeat_index].unsqueeze(0)
        states = states_single.repeat(num_samples, 1, 1)

        actions_single = actions[repeat_index].unsqueeze(0)
        actions = actions_single.repeat(num_samples, 1, 1)

    states = states.transpose(0, 1)
    actions = actions.transpose(0, 1)

    losses = []
    y = labels_dict["copulation"]
    with torch.no_grad():
        for k in range(3):
            env.reset(init_state=states[0].clone())
            model.reset_policy(labels=y, temperature=args.temperature)

            rollout_states, rollout_actions = generate_rollout(env, model, burn_in=args.burn_in,
                                                               burn_in_actions=actions, horizon=actions.size(0))
            rollout_states = rollout_states.transpose(0, 1)
            rollout_actions = rollout_actions.transpose(0, 1)


            # if we have a single agent setting, we generate two rollouts and vert stack them
            if bad_experiment:
                rollout_states_2, rollout_actions_2 = generate_rollout(env, model, burn_in=args.burn_in,
                                                                   burn_in_actions=actions, horizon=actions.size(0))
                rollout_states_2 = rollout_states_2.transpose(0, 1)
                rollout_actions_2 = rollout_actions_2.transpose(0, 1)

                stack_tensor_states = torch.cat((rollout_states, rollout_states_2), dim=2)
                stack_tensor_action = torch.cat((rollout_actions, rollout_actions_2), dim=2)

                rollout_states_3, rollout_actions_3 = generate_rollout(env, model, burn_in=args.burn_in,
                                                                   burn_in_actions=actions, horizon=actions.size(0))
                rollout_states_3 = rollout_states_3.transpose(0, 1)
                rollout_actions_3 = rollout_actions_3.transpose(0, 1)

                rollout_states_4, rollout_actions_4 = generate_rollout(env, model, burn_in=args.burn_in,
                                                                   burn_in_actions=actions, horizon=actions.size(0))
                rollout_states_4 = rollout_states_4.transpose(0, 1)
                rollout_actions_4 = rollout_actions_4.transpose(0, 1)

                stack_tensor_states_2 = torch.cat((rollout_states_3, rollout_states_4), dim=2)
                stack_tensor_action_2 = torch.cat((rollout_actions_3, rollout_actions_4), dim=2)

                final_states_tensor = torch.cat((stack_tensor_states, stack_tensor_states_2), dim=1)
                final_actions_tensor = torch.cat((stack_tensor_action, stack_tensor_action_2), dim=1)

                losses.append(get_classification_loss(final_states_tensor, final_actions_tensor))

            else:
                losses.append(get_classification_loss(rollout_states, rollout_actions))

    print(np.mean(losses))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--exp_folder', type=str,
                        required=True, default=None,
                        help='folder of experiments from which to load models')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('-n', '--num_samples', type=int,
                        required=False, default=8,
                        help='number of samples to generate FOR EACH CLASS')
    parser.add_argument('-v', '--num_values', type=int,
                        required=False, default=3,
                        help='number of values to evaluate for continuous LFs')
    parser.add_argument('-r', '--repeat_index', type=int,
                        required=False, default=-1,
                        help='repeated sampling with same burn-in')
    parser.add_argument('-b', '--burn_in', type=int,
                        required=False, default=0,
                        help='burn in period, for sequential data')
    parser.add_argument('-t', '--temperature', type=float,
                        required=False, default=1.0,
                        help='sampling temperature')
    args = parser.parse_args()

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    print(os.path.join(exp_dir, 'master.json'))
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    assert args.repeat_index < args.num_samples
    if args.repeat_index >= 0:
        assert args.burn_in > 0

    # Check self consistency
    for trial_id in master['summaries']:
        visualize_samples_ctvae(exp_dir, trial_id, args.num_samples, args.num_values, args.repeat_index, args.burn_in,
                                args.temperature, bad_experiment=False)
