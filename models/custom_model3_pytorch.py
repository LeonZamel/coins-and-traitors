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

class CustomComplexInputNetwork3Torch(TorchModelV2, nn.Module):
    """
    Custom model for the coins traitors voting environment.
    Inputs are:
    - a general map view with walls, coins, and broken coins
    - a map view containing the observable agents
    - the current votes as a AxA matrix
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

        ### MAP and AGENTS CONV ###
        self.conv_module_out_channels = 32

        self.map_and_agents_conv_module = nn.Sequential(
            nn.Conv3d(
                map_channels+1,
                16,
                (1,3,3),
                padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(
                16,
                32, 
                (1,3,3),
                padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(
                32,
                self.conv_module_out_channels, 
                (1,3,3),
                padding=(0,1,1)),
            nn.ReLU(),
        )

        ### VOTES POOLING ###
        self.votes_avg_pool = nn.AvgPool2d((1, num_a))
        self.votes_max_pool = nn.MaxPool2d((1, num_a))

        ### FC NET ###
        # This net behaves like an fc-net with shared layers. By using 1dConv
        # along the channels we have the fc structure, along the other 1d dim we have the different agents
        self.fc_module = nn.Sequential(
            nn.Conv1d(
                self.conv_module_out_channels*map_height*map_width + 2 + 2 + 1 + 1 + (1 if self.traitor else 0),
                256,
                (1)),
            nn.ReLU(),
            nn.Conv1d(
                256,
                256, 
                (1)),
            nn.ReLU(),
        )

        ### VOTES NET ###
        self.votes_module = nn.Sequential(
            nn.Conv1d(
                256,
                128,
                (1)),
            nn.ReLU(),
            nn.Conv1d(
                128,
                2, 
                (1)),
            nn.ReLU(),
        )

        ### MOVE NET ###
        self.move_module = nn.Sequential(
            nn.Linear(
                256,
                128),
            nn.ReLU(),
            nn.Linear(
                128,
                5),
            nn.ReLU(),
        )

        ### VALUE NET ###
        self.value_module = nn.Sequential(
            nn.Linear(
                256,
                128),
            nn.ReLU(),
            nn.Linear(
                128,
                1),
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

        votes_T = votes.transpose(2,3)
        # shape (batch, 1, num_agents, num_agents)

        # We transpose the votes once and stack them onto the other votes so we have 2 channels
        votes = torch.cat([votes, votes_T], dim=1)
        # shape (batch, 2, num_agents, num_agents)


        ### GENERAL MAP and AGENTS MAP ###
        # The general map has shape (batchsize, viewsize, viewsize, channels)
        map_obs = orig_obs['map']

        # The agents obs (again, similar to votes) is a 2D Python array with shape (viewsize, viewsize, batchsize, num_agents+1) as
        # [[<Tensor, shape (batch, num_agents+1)>, ...], ...]
        # The reason we have num_agents+1 is because we need a value for "no agent here". One hot encoded this will be the last layer, which we will simply ignore
        map_agents_obs = convert_to_tensor(orig_obs['map_agents'])
        # We first transpose to (num_agents+1, batchsize, viewsize, viewsize)
        map_agents_obs = map_agents_obs.permute(3,2,0,1)
        # shape (num_agents+1, batchsize, viewsize, viewsize)

        # We can now concatenate each agent view with the map view
        map_and_agents_obs = []
        for i in range(self.num_agents):
            a_obs = torch.unsqueeze(map_agents_obs[i], 3)
            # shape (batch, viewsize, viewsize, 1)
            m_and_a_obs = torch.cat([map_obs, a_obs], dim=3).permute(0,3,1,2)
            # shape (batch, channels+1, viewsize, viewsize)
            map_and_agents_obs.append(m_and_a_obs)

        map_and_agents = torch.stack(map_and_agents_obs, dim=2)
        # shape (batch, channels+1, num_agents, view_size, view_size)
        # Done, we can do a 3d conv with kernel size 1 for the num_agents dim (kernelsize (1,x,x))

        ### PLAYING ###
        # Python array of shape (numagents, batch, 2) as [<Tensor, shape (batch, 2)>, ...]
        playing_obs = convert_to_tensor(orig_obs['playing'])
        playing_obs = playing_obs.permute(1,0,2)
        # Similar to the votes above this is one hot encoded. We slice form 1:2 so the dimension of size 1 stays
        # we keep it for the convolution in the net
        playing = playing_obs[:,:,1:2]
        # shape (batch, num_agents, 1)
        playing = playing.transpose(1,2)
        # shape (batch, 1, num_agents)

        ### OWN ID ###
        own_id = torch.unsqueeze(orig_obs['own_id'], dim=2)
        # shape (batch, num_agents, 1)
        own_id = own_id.transpose(1,2)
        # shape (batch, 1, num_agents)

        ### TRAITORS ###
        traitors = None
        if self.traitor:
            traitors_obs = convert_to_tensor(orig_obs['traitors'])
            # shape (num_agents, batch, 2)
            # Last dim is one-hot-encoded again. Only use second value, slice to keep dim
            traitors_obs = traitors_obs[:,:,1:2]
            traitors = traitors_obs.permute(1,2,0)
            # shape (batch, 1, num_agents)
        ## Input processing done ##


        ##### RUN VALUES THROUGH MODULES #####
        m_and_a_out = self.map_and_agents_conv_module(map_and_agents)
        # shape (batch, self.conv_module_out_channels, numagents, viewsize, viewsize)
        m_and_a_out = m_and_a_out.transpose(1,2)
        # shape (batch, numagents, self.conv_module_out_channels, viewsize, viewsize)
        m_and_a_out = torch.flatten(m_and_a_out, start_dim=2)
        # shape (batch, numagents, self.conv_module_out_channels * viewsize * viewsize)
        m_and_a_out = m_and_a_out.transpose(1,2)
        # shape (batch, self.conv_module_out_channels * viewsize * viewsize, numagents)

        votes_avg_out = torch.squeeze(self.votes_avg_pool(votes), dim=3)
        votes_max_out = torch.squeeze(self.votes_max_pool(votes), dim=3)
        # shape (batch, 2, numagents), before squeezing it was (batch, 2, numagents, 1)

        # print(m_and_a_out.shape)
        # print(votes_avg_out.shape)
        # print(votes_max_out.shape)
        # print(playing.shape)
        # print(own_id.shape)
        # if self.traitor:
        #     print(traitors.shape)
        
        concatenated = torch.cat([m_and_a_out, votes_avg_out, votes_max_out, playing, own_id] + ([traitors] if self.traitor else []), dim=1)
        # shape (batch, self.conv_module_out_channels * viewsize * viewsize + 2 + 2 + 1 + 1 + (1 if self.traitor else 0), numagents)

        hidden = self.fc_module(concatenated)
        # shape (batch, 256, numagents)

        vote_logits = self.votes_module(hidden)
        # shape (batch, 2, numagents)
        vote_logits = vote_logits.transpose(1,2).reshape([-1, self.num_agents*2])
        # shape (batch, numagents * 2)

        sum_hidden = torch.sum(hidden, dim=2)
        move_logits = self.move_module(sum_hidden)
        # shape (batch, 5)

        logits = torch.cat([move_logits, vote_logits], dim=1)

        self._value_out = self.value_module(sum_hidden)

        return logits, []

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self._value_out, [-1])
