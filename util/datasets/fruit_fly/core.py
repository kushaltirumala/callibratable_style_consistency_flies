import os
import numpy as np
import torch

from util.datasets import TrajectoryDataset
from .label_functions import label_functions_list

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage.transform import resize


# CONSTANTS
LENGTH = 94
WIDTH = 50
SCALE = 10

COORDS = {
    'first_fly' : [0,1],
    'second_fly' : [2,3]
}

# TODO let users define where data lies
# ROOT_DIR = 'transforming_data/compressed_final_data/'
ROOT_DIR = '/cs/courses/cs101a/segmented_fruit_fly_data/'
TRAIN_FILE = 'copulation_segmented_train.npz'
TEST_FILE = 'copulation_segmented_test.npz'
TRAIN_LABELS = 'copulation_segmented_train_label.npz'
TEST_LABELS = 'copulation_segmented_test_label.npz'



class FruitFlyDataset(TrajectoryDataset):

    name = 'fruit_fly'
    all_label_functions = label_functions_list

    # Default config
    _seq_len = 100
    _state_dim = 4
    _action_dim = 4

    normalize_data = True
    single_agent = False
    player_types = {
        'first_fly' : True,
        'second_fly' : True,
    }

    def __init__(self, data_config):
        super().__init__(data_config)

    def _load_data(self):
        # Process configs
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']
        if 'single_agent' in self.config:
            self.single_agent = self.config['single_agent']
        if 'player_types' in self.config:
            self.player_types = self.config['player_types']
        if 'discard_env_agent' in self.config:
            self.discard_env_agent = self.config['discard_env_agent']

        # TODO hacky solution
        if 'labels' in self.config:
            for lf_config in self.config['labels']:
                lf_config['data_normalized'] = self.normalize_data


        self.train_states, self.train_actions, self.train_labels = self._load_and_preprocess(train=True)
        self.test_states, self.test_actions, self.test_labels = self._load_and_preprocess(train=False)
        self.hasTrueLabels = True


    def _load_and_preprocess(self, train):
        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else TEST_FILE)
        labels_path = os.path.join(ROOT_DIR, TRAIN_LABELS if train else TEST_LABELS)

        file = np.load(path)
        labels_file = np.load(labels_path)

        data = file['data']
        labels = labels_file['data']

        # Subsample timesteps
        data = data[:,::self.subsample]
        # print("HERE")
        # print(data.shape)


        # Filter based on player types
        # inds_filter = []
        # for key, val in self.player_types.items():
        #     inds_filter += COORDS[key] if val else []
        # # print(inds_filter)
        # data = data[:,:,inds_filter]

        # print(data.shape)



        # Normalize data
        if self.normalize_data:
            data = normalize(data)

        # Convert to single-agent data
        if self.discard_env_agent:
            # if we want to run single agent learning with discarding one fly
            data = data[:, :, 2:4]
            labels = labels[:, 1:]
        elif self.single_agent:
            seq_len = data.shape[1]
            data = np.swapaxes(data, 0, 1)
            data = np.reshape(data, (seq_len, -1, 2))
            data = np.swapaxes(data, 0, 1)


            labels = np.reshape(labels, (-1, 1))

            # also fix the corresponding labels


        # Convert to states and actions
        states = data
        actions = states[:,1:] - states[:,:-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]


        if train:
            self.train_data = data
        else:
            self.test_data = data

        # print("HEREHERE")
        # print(states.shape)

        return torch.Tensor(states), torch.Tensor(actions), torch.Tensor(labels)

    def save(self, states, actions=[], save_path='', save_name='', burn_in=0, labels=None, lf_list=[], single_plot=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        states = states.detach().numpy()
        if self.normalize_data:
            states = unnormalize(states)
        #
        # if single_plot:
        #     states = np.swapaxes(states, 0, 1)
        #     states = np.reshape(states, (states.shape[0], 1, -1))
        #     states = np.swapaxes(states, 0, 1)

        # n_players = int(states.shape[-1]/2)
        # colormap = ['b'] * n_players
        # colormap = ['b'] * n_players
        # colormap = ['b', 'g', 'r', 'm', 'y'] # TODO colormap for more players

        # for i in range(len(states)):
        #     seq = SCALE*states[i]
        #
        #     fig, ax = _set_figax()
        #
            # for k in range(n_players):
            #     x = seq[:,(2*k)]
            #     y = seq[:,(2*k+1)]
            #     c = colormap[k]
            #
            #     # if not single_plot:
            #     # ax.plot(x, y, 'o', color=c, markersize=(4 if single_plot else 8), alpha=0.5)
            #     ax.plot(x, y, 'o', color=c, markersize=5, alpha=0.5)
            #     ax.plot(x, y, color=c, linewidth=5, alpha=0.7) # DEFAULT lw=3, a=0.7
            #     ax.plot(x, y, color='k', linewidth=1, alpha=0.7) # DEFAULT lw=3, a=0.7
            #
            # # Starting positions
            # x = seq[0,::2]
            # y = seq[0,1::2]
            # ax.plot(x, y, 'd', color='black', markersize=16) # DEFAULT 'O', 10
            #
            # # Final positions
            # x = seq[-1,::2]
            # y = seq[-1,1::2]
            # ax.plot(x, y, 'o', color='black', markersize=8)
        #
        #     # Burn-ins
        #     if burn_in > 0:
        #         x = seq[:burn_in,0] if single_plot else seq[:burn_in,::2]
        #         y = seq[:burn_in,1] if single_plot else seq[:burn_in,1::2]
        #
        #         ax.plot(x, y, color='black', linewidth=8, alpha=0.5)
        #
        #     # (Optional) Label function plotting
        #     for lf in lf_list:
        #         label = labels[i].numpy()
        #         ax = lf.plot(ax, seq, label, width=WIDTH*SCALE, length=LENGTH*SCALE)
        #     print(states)
            # states is 8 long, each one is size 50
            n_players = 2
            colors = ['r', 'b']
            for i in range(len(states)):
                # fig, ax = _set_figax()
                seq = states[i]
                for k in range(n_players):
                    x = seq[:, (2 * k)]
                    y = seq[:, (2 * k + 1)]
                    # c = colormap[k]

                    # if not single_plot:
                    # ax.plot(x, y, 'o', color=c, markersize=(4 if single_plot else 8), alpha=0.5)
                    plt.plot(x, y, 'o', color=colors[k], markersize=5, alpha=0.5)
                    # ax.plot(x, y, color=c, linewidth=5, alpha=0.7)  # DEFAULT lw=3, a=0.7
                    # ax.plot(x, y, color='k', linewidth=1, alpha=0.7)  # DEFAULT lw=3, a=0.7

                # Starting positions

                for k in range(n_players):
                    x = seq[0, (2*k)]
                    y = seq[0, (2*k + 1)]
                    plt.plot(x, y, 'd', color=colors[k], markersize=8)  # DEFAULT 'O', 10

                    # Final positions
                    x = seq[-1, (2*k)]
                    y = seq[-1, (2*k + 1)]
                    plt.plot(x, y, 'o', color=colors[k], markersize=8)


                # plt.tight_layout(pad=0)
                plt_title = "copulation: "
                if labels is not None:
                    if labels[i][0] == 1:
                        # this means copulation is occurring (i.e label is [1, 0] for trajectory
                        plt_title += " true"
                    else:
                        plt_title += "false"

                plt.title(plt_title)

                if len(save_name) == 0:
                    plt.savefig(os.path.join(save_path, '{:03d}.png'.format(i)))
                else:
                    plt.savefig(os.path.join(save_path, '{}.png'.format(save_name)))

                plt.close()

def normalize(data):
    """Scale by dimensions of court and mean-shift to center of left half-court."""
    state_dim = data.shape[2]
    shift = [int(WIDTH/2)] * state_dim
    scale = [LENGTH, WIDTH] * int(state_dim/2)
    return np.divide(data-shift, scale)

def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2]
    shift = [int(WIDTH/2)] * state_dim
    scale = [LENGTH, WIDTH] * int(state_dim/2)
    return np.multiply(data, scale) + shift

def _set_figax():
    fig = plt.figure(figsize=(5,5))
    # img = plt.imread(os.path.join(ROOT_DIR, 'court.png'))
    # img = resize(img,(SCALE*WIDTH,SCALE*LENGTH,3))

    ax = fig.add_subplot(111)
    # ax.imshow(img)

    # Show just the left half-court
    ax.set_xlim([-500,550])
    ax.set_ylim([-500,550])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax
