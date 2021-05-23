# ADAPTED FROM 
# https://github.com/ray-project/ray/blob/master/rllib/models/torch/complex_input_net.py

from gym.spaces import Box, Discrete, Tuple
import numpy as np
from ray.rllib.models.torch.misc import normc_initializer as \
    torch_normc_initializer, SlimFC
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import one_hot
from ray.rllib.policy.sample_batch import SampleBatch

import torch
import torch.nn as nn

from .util import convert_to_tensor

class PermEquivariantModelLarge(TorchModelV2, nn.Module):
    """
    Custom model for the coins traitors voting environment.
    Inputs are:
    - a general map view with walls, coins, and broken coins
    - a map view containing the observable agents
    - the current votes as a PxP matrix
    - which agents are still in the game (i.e. not removed)
    - the own id
    and, if they are a traitor
    - who the other traitors are

    The model is equivariant under permutations of agents.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        
        self.original_space = obs_space.original_space

        self.traitor = kwargs['traitor']

        self.num_agents = len(self.original_space['votes'])
        num_a = self.num_agents
        map_height, map_width, map_channels = self.original_space['map'].shape
        assert map_height == map_width, "Observable map height and width should be the same"

        self.map_size = map_height

        self.hidden_channels = 32

        # Shape is indicated in variable names:
        # P = num_agents (population size)
        # W = world size = map_size = map_height = map_width,
        # C = map_channels
        # X = hidden_channels
        # First number indicates number of channels
        # e.g. votes_avg_pool_1_P_P__1_P_1 is an average pooling operation 
        # that takes a PxP matrix with 1 channel to a 1xP matrix with 1 channel

        ### MAP GENERAL ###
        self.map_general_C_W_W__X_W_W = nn.Conv2d(map_channels, self.hidden_channels, 1)

        ### MAP AGENTS
        self.map_agents_first_1_P_W_W__X_P_W_W = nn.Conv3d(1, self.hidden_channels, 1)
        self.map_agents_second_1_P_W_W__X_P_W_W = nn.Conv3d(1, self.hidden_channels, 1)

        ### VOTES ###
        # Starts of as 1_P_P
        self.votes_conv_first_1_P_P__X_P_P = nn.Conv2d(1,self.hidden_channels,1)
        self.votes_conv_second_1_P_P__X_P_P = nn.Conv2d(1,self.hidden_channels,1)

        # Pool along both dims
        self.votes_avg_pool_1_P_P__1_P_1 = nn.AdaptiveAvgPool2d((num_a, 1))
        self.votes_max_pool_1_P_P__1_P_1 = nn.AdaptiveMaxPool2d((num_a, 1))
        self.votes_avg_pool_1_P_P__1_1_P = nn.AdaptiveAvgPool2d((1, num_a))
        self.votes_max_pool_1_P_P__1_1_P = nn.AdaptiveMaxPool2d((1, num_a))

        # Then apply conv along both
        self.votes_conv_avg_1_P_1__X_P_1 = nn.Conv2d(1,self.hidden_channels,1)
        self.votes_conv_max_1_P_1__X_P_1 = nn.Conv2d(1,self.hidden_channels,1)
        self.votes_conv_avg_1_1_P__X_1_P = nn.Conv2d(1,self.hidden_channels,1)
        self.votes_conv_max_1_1_P__X_1_P = nn.Conv2d(1,self.hidden_channels,1)

        ### PLAYING ###
        self.playing_conv_1_1_P__X_1_P = nn.Conv2d(1,self.hidden_channels,1)
        self.playing_conv_1_P_1__X_P_1 = nn.Conv2d(1,self.hidden_channels,1)

        ### OWN ID ###
        self.own_id_conv_1_1_P__X_1_P = nn.Conv2d(1,self.hidden_channels,1)
        self.own_id_conv_1_P_1__X_P_1 = nn.Conv2d(1,self.hidden_channels,1)

        ### TRAITORS ###
        self.traitors_conv_1_1_P__X_1_P = nn.Conv2d(1,self.hidden_channels,1)
        self.traitors_conv_1_P_1__X_P_1 = nn.Conv2d(1,self.hidden_channels,1)

        ### AGG ###
        self.activation = nn.ReLU()

        ### FINAL ###
        # Intermediary
        self.intermediary = nn.Sequential(
            nn.Conv3d(
                self.hidden_channels * 4,
                self.hidden_channels * 2,
                (1,3,3),
                padding=(0,1,1)),
            nn.MaxPool3d(
                (1,3,3),
                (1,2,2)),
            nn.Conv3d(
                self.hidden_channels * 2,
                self.hidden_channels, 
                (1,3,3),
                padding=(0,1,1)),
            nn.MaxPool3d(
                (1,3,3),
                (1,1,1)),
        )

        # Votes
        self.votes_net = nn.Sequential(
            nn.Conv1d(self.hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 2, 1),
            nn.ReLU(),
        )

        # Moves
        self.moves_net = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels * 4,
                self.hidden_channels * 2,
                3,
                padding=1),
            nn.MaxPool2d(
                3,
                2),
            nn.Conv2d(
                self.hidden_channels * 2,
                self.hidden_channels, 
                3,
                padding=1),
            nn.MaxPool2d(
                3,
                1),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.ReLU(),
        )

        # Value
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )
 
        self._value_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and 'obs_flat' in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS],
                                                   self.obs_space, 'torch')

        ###### Process inputs #####

        ### GENERAL MAP and AGENTS MAP ###
        # The general map has shape (batchsize, viewsize, viewsize, channels)
        map_obs = orig_obs['map']
        map_obs = map_obs.permute(0,3,1,2)

        # The agents obs (again, similar to votes) is a 2D Python array with shape (viewsize, viewsize, batchsize, num_agents+1) as
        # [[<Tensor, shape (batch, num_agents+1)>, ...], ...]
        # The reason we have num_agents+1 is because we need a value for "no agent here". One hot encoded this will be the last layer, which we will simply slice away
        map_agents_obs = convert_to_tensor(orig_obs['map_agents'])
        # We first transpose to (num_agents+1, batchsize, viewsize, viewsize)
        map_agents_obs = map_agents_obs.permute(2,3,0,1)
        # shape (batchsize, num_agents+1, viewsize, viewsize)
        # Remove last layer form agents
        map_agents_obs = map_agents_obs[:,:-1,:,:]
        # shape (batchsize, num_agents, viewsize, viewsize)

        # Add a dummy channel
        map_agents_obs = map_agents_obs.unsqueeze(1)
        # shape (batchsize, 1, num_agents, viewsize, viewsize)

        ### VOTES ###
        # We get a 2D Python array like [[<Tensor, shape (?,2)>, ...], ...], where '?' is the batch size
        # so we need to first convert this to a tensor of shape (num_agent, num_agents, batchsize, 2)
        # and then pull out the batch dimension to the front so we get (batchsize, num_agent, num_agents, 2) tensors as votes 

        votes_obs = convert_to_tensor(orig_obs['votes']).permute(2,0,1,3)
        # The votes are one hot encoded, so (1,0) means no vote and (0,1) means vote
        # We can just take the second component by indexing with 1:2. We slice to keep that dim for the conv later
        votes = votes_obs[:,:,:,1:2]
        # shape (batch, num_agents, num_agents, 1)
        # Push channels dim to correct position
        votes = votes.permute(0,3,1,2)
        # shape (batch, 1, num_agents, num_agents)

        ### PLAYING ###
        # Python array of shape (numagents, batch, 2) as [<Tensor, shape (batch, 2)>, ...]
        playing_obs = convert_to_tensor(orig_obs['playing'])
        playing_obs = playing_obs.permute(1,0,2)
        # Similar to the votes above this is one hot encoded. We slice form 1:2 so the dimension of size 1 stays
        # we keep it for the convolution in the net
        playing = playing_obs[:,:,1:2]
        # shape (batch, num_agents, 1)
        playing = torch.unsqueeze(playing.transpose(1,2), dim=1)
        # shape (batch, 1, 1, num_agents)
        playing_T = playing.transpose(2,3)
        # shape (batch, 1, num_agents, 1)

        ### OWN ID ###
        own_id = torch.unsqueeze(orig_obs['own_id'], dim=2)
        # shape (batch, num_agents, 1)
        own_id = torch.unsqueeze(own_id.transpose(1,2), dim=1)
        # shape (batch, 1, 1, num_agents)
        own_id_T = own_id.transpose(2,3)
        # shape (batch, 1, 1, num_agents)

        ### TRAITORS ###
        traitors = None
        traitors_T = None
        if self.traitor:
            traitors_obs = convert_to_tensor(orig_obs['traitors'])
            # shape (num_agents, batch, 2)
            # Last dim is one-hot-encoded again. Only use second value, slice to keep dim
            traitors_obs = traitors_obs[:,:,1:2]
            traitors = torch.unsqueeze(traitors_obs.permute(1,2,0), dim=1)
            # shape (batch, 1, 1, num_agents)
            traitors_T = traitors.transpose(2,3)
            # shape (batch, 1, num_agents, 1)
        ## Input processing done ##


        ##### RUN VALUES THROUGH MODULES #####
        map_X_1_1_W_W = self.map_general_C_W_W__X_W_W(map_obs).unsqueeze(2).unsqueeze(2)

        map_agents_X_1_P_W_W = self.map_agents_first_1_P_W_W__X_P_W_W(map_agents_obs).unsqueeze(2)
        map_agents_X_P_1_W_W = self.map_agents_second_1_P_W_W__X_P_W_W(map_agents_obs).unsqueeze(2).transpose(2, 3)

        x = map_X_1_1_W_W

        x = torch.add(x, map_agents_X_1_P_W_W)
        x = torch.add(x, map_agents_X_P_1_W_W)

        x = torch.add(x, self.votes_conv_first_1_P_P__X_P_P(votes).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.votes_conv_second_1_P_P__X_P_P(votes).transpose(2,3).unsqueeze(4).unsqueeze(4))
        
        avg_pool_1_P_1 = self.votes_avg_pool_1_P_P__1_P_1(votes)
        max_pool_1_P_1 = self.votes_max_pool_1_P_P__1_P_1(votes)
        avg_pool_1_1_P = self.votes_avg_pool_1_P_P__1_1_P(votes)
        max_pool_1_1_P = self.votes_max_pool_1_P_P__1_1_P(votes)
        
        x = torch.add(x, self.votes_conv_avg_1_P_1__X_P_1(avg_pool_1_P_1).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.votes_conv_max_1_P_1__X_P_1(max_pool_1_P_1).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.votes_conv_avg_1_1_P__X_1_P(avg_pool_1_1_P).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.votes_conv_max_1_1_P__X_1_P(max_pool_1_1_P).unsqueeze(4).unsqueeze(4))

        x = torch.add(x, self.playing_conv_1_1_P__X_1_P(playing).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.playing_conv_1_P_1__X_P_1(playing_T).unsqueeze(4).unsqueeze(4))
        
        x = torch.add(x, self.own_id_conv_1_1_P__X_1_P(own_id).unsqueeze(4).unsqueeze(4))
        x = torch.add(x, self.own_id_conv_1_P_1__X_P_1(own_id_T).unsqueeze(4).unsqueeze(4))

        if self.traitor:
            ### TRAITORS ###
            x = torch.add(x, self.traitors_conv_1_1_P__X_1_P(traitors).unsqueeze(4).unsqueeze(4))
            x = torch.add(x, self.traitors_conv_1_P_1__X_P_1(traitors_T).unsqueeze(4).unsqueeze(4))
        
        x = self.activation(x)
        # shape (batch, self.hidden_channels, num_agents, num_agents, world_size, world_size)

        avg_pool_first_X_P_W_W = torch.mean(x, 3)
        max_pool_second_X_P_W_W, _ = torch.max(x, 3) # Max function also returns indices when called with dim argument
        avg_pool_first_X_P_W_W = torch.mean(x, 2)
        max_pool_second_X_P_W_W, _ = torch.max(x, 2)

        concat = torch.cat([avg_pool_first_X_P_W_W, max_pool_second_X_P_W_W, avg_pool_first_X_P_W_W, max_pool_second_X_P_W_W], dim=1)
        # shape (batch, 4 * self.hidden_channels, numagents, world_size, world_size)

        intermediary = self.intermediary(concat)
        # shape (batch, self.hidden_channels, numagents, 1, 1)

        intermediary = intermediary.squeeze(dim=3).squeeze(dim=3)
        # shape (batch, self.hidden_channels, numagents)

        vote_logits = self.votes_net(intermediary)
        # shape (batch, 2, numagents)

        vote_logits = vote_logits.transpose(1,2).reshape([-1, self.num_agents*2])
        # shape (batch, numagents * 2)

        sum_hidden = torch.sum(concat, dim=2)
        # shape (batch, 4 * self.hidden_channels, world_size, world_size)

        move_logits = self.moves_net(sum_hidden)
        # shape (batch, 5)

        logits = torch.cat([move_logits, vote_logits], dim=1)

        intermediary_sum = torch.sum(intermediary, dim=2)
        self._value_out = self.value_net(intermediary_sum)

        return logits, []

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self._value_out, [-1])

