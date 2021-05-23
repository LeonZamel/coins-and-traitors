# ADAPTED FROM 
# https://github.com/ray-project/ray/blob/master/rllib/models/tf/complex_input_net.py

import numpy as np

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import one_hot
from tensorflow import keras

tf1, tf, tfv = try_import_tf()


class CustomComplexInputNetwork3(TFModelV2):
    """
    Custom model for the coins traitors voting environment.
    Inputs are:
    - a general map view with walls, coins, and broken coins
    - a mapo view containing the observable agents
    - the current votes as a AxA matrix
    - which agents are still in the game (i.e. not removed)
    - the own id
    and, if they are a traitor
    - who the other traitors are

    The model is equivariant under permutations of agents.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        
        self.original_space = obs_space.original_space

        self.traitor = kwargs['traitor']

        self.num_agents = len(self.original_space['votes'])
        map_height, map_width, map_channels = self.original_space['map'].shape

        ### MAIN MODEL ###
        # W x W x C+1
        map_and_agents_input = tf.keras.layers.Input(shape=(map_height, map_width, map_channels+1), name=f'map_and_agents_observation')
        # P x 2
        votes_input = tf.keras.layers.Input(shape=(self.num_agents, 2), name='votes_observations')
        # P x 1
        playing_input = tf.keras.layers.Input(shape=(self.num_agents, 1), name='playing_observations')
        # 1
        own_id_input = tf.keras.layers.Input(shape=(1, ), name='own_id_observation')

        traitor_input = None
        if self.traitor:
            # The input if the current 
            # 1
            traitor_input = tf.keras.layers.Input(shape=(1, ), name='traitors_observation')
        
        # Map pre-processing
        last = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')(map_and_agents_input)
        last = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')(last)
        last = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')(last)
        map_and_agents = tf.keras.layers.Flatten()(last)

        # Votes pre-processing
        max_pool = tf.keras.layers.GlobalMaxPool1D()(votes_input)
        max_pool = tf.keras.layers.Flatten()(max_pool)

        avg_pool = tf.keras.layers.GlobalAvgPool1D()(votes_input)
        avg_pool = tf.keras.layers.Flatten()(avg_pool)

        # Playing pre-processing
        playing_avg = tf.keras.layers.GlobalAvgPool1D()(playing_input)
        playing_avg = tf.keras.layers.Flatten()(playing_avg)

        concat = tf.keras.layers.Concatenate(axis=1)([map_and_agents, max_pool, avg_pool, playing_avg, own_id_input] + ([traitor_input] if self.traitor else []))

        vote_fc = tf.keras.layers.Dense(
            256,
            activation='relu')(concat)
        vote_fc = tf.keras.layers.Dense(
            256,
            activation='relu')(vote_fc)
        vote_output = tf.keras.layers.Dense(
            2,
            activation='relu',
            name='vote_logits')(vote_fc)
        
        vote_fc = tf.keras.layers.Dense(
            256,
            activation='relu')(concat)
        vote_fc = tf.keras.layers.Dense(
            256,
            activation='relu')(vote_fc)

        output_channels = 32
        
        general_output = tf.keras.layers.Dense(
            output_channels,
            activation='relu',
            name='general_out')(vote_fc)

        self.main_model = tf.keras.models.Model(
            [map_and_agents_input, votes_input, playing_input, own_id_input] + ([traitor_input] if self.traitor else []), 
            [general_output, vote_output])


        ### MOVE AND VALUE MODEL ###
        # P x Y
        general_unbatched_input = tf.keras.layers.Input(shape=(self.num_agents, output_channels, ), name='general_input')

        layer_add = tf.math.reduce_sum(general_unbatched_input, axis=1)
        layer_avg = tf.keras.layers.GlobalAvgPool1D()(general_unbatched_input)
        layer_max = tf.keras.layers.GlobalMaxPool1D()(general_unbatched_input)

        concat = tf.keras.layers.Concatenate(axis=1)([layer_add, layer_avg, layer_max])

        # Move logits branch
        move_logits = tf.keras.layers.Dense(
            256,
            activation='relu')(concat)
        move_logits = tf.keras.layers.Dense(
            256,
            activation='relu')(move_logits)
        move_logits = tf.keras.layers.Dense(
            5,
            activation='relu')(move_logits)
        
        # Value branch
        val_branch = tf.keras.layers.Dense(
            128,
            activation='relu')(concat)
        val_out = tf.keras.layers.Dense(
            1,
            name='value_out',
            activation=None,
            kernel_initializer=normc_initializer(0.01))(val_branch)

        self.move_value_model = tf.keras.models.Model(
            general_unbatched_input, 
            [move_logits, val_out])

        self._value_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and 'obs_flat' in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS],
                                                   self.obs_space, 'tf')

        ###### Process inputs #####

        ### VOTES ###
        # We get a 2D Python array like [[<Tensor, shape (?,2)>, ...], ...], where '?' is the batch size
        # so we need to first convert this to a tensor of shape (num_agent, num_agents, batchsize, 2)
        # and then pull out the batch dimension to the front so we get (batchsize, num_agent, num_agents, 2) tensors as votes 
        votes_obs = tf.transpose(tf.convert_to_tensor(orig_obs['votes']), [2,0,1,3])
        # The votes are one hot encoded, so (1,0) means no vote and (0,1) means vote
        # We can just take the second component by slicing from 1:2 and not simply indexing we keep the dim of size 1
        # we can then stack along that dimension next
        votes_obs = votes_obs[:,:,:,1:2]
        # shape (batch, num_agents, num_agents)
        # We also take the transpose of this matrix
        votes_obs_T = tf.transpose(votes_obs, [0,2,1,3])
        # Stack them so we have two channels
        votes = tf.concat([votes_obs, votes_obs_T], axis=3)
        votes = self.combine_batch_and_num_agents_dim(votes)

        ### GENERAL MAP and AGENTS MAP ###
        # The general map has shape (batchsize, viewsize, viewsize, channels)
        map_obs = orig_obs['map']

        # The agents obs (again, similar to votes) is a 2D Python array with shape (viewsize, viewsize, batchsize, num_agents+1)
        # The reason we have num_agents+1 is because we need a value for "no agent here". One hot encoded this will be the last layer, which we will simply ignore
        map_agents_obs = tf.convert_to_tensor(orig_obs['map_agents'])
        # We first transpose to (num_agents+1, batchsize, viewsize, viewsize)
        map_agents_obs = tf.transpose(map_agents_obs, [3,2,0,1])

        # We can now concatenate each agent view with the map view
        map_and_agents_obs = []
        for i in range(self.num_agents):
            a_obs = tf.expand_dims(map_agents_obs[i], 3)
            map_and_agents_obs.append(tf.concat([map_obs, a_obs], axis=3))

        map_and_agents_obs = tf.convert_to_tensor(map_and_agents_obs)
        # shape (num_agents, batch, view_size, view_size, channels+1)
        # Flip first two axes so we can batch over the num_agents
        map_and_agents_obs = tf.transpose(map_and_agents_obs, [1,0,2,3,4])
        # shape (batch, num_agents, view_size, view_size, channels+1)
        # Pull num_agents dim into batch
        map_and_agents = self.combine_batch_and_num_agents_dim(map_and_agents_obs)
        # shape (num_agents * batch, view_size, view_size, channels+1)

        ### PLAYING ###
        playing_obs = tf.convert_to_tensor(orig_obs['playing'])
        playing_obs = tf.transpose(playing_obs, [1,0,2])
        # Similar to the votes above this is one hot encoded. However, we instead slice form 1:2 so the dimension of size 1 stays
        # we keep it for the 1D convolution in the net
        playing = playing_obs[:,:,1:2]
        # shape (batch, num_agents, 1)
        # We do not pull the num_agents into the batch dimension here, since we want to aggregate this as a whole
        # however, we must replicate the vector num_agents times to get to the correct batchsize
        playing = tf.stack([playing for _ in range(self.num_agents)], axis=1)
        # shape (batch, num_agents, num_agents, 1)
        # Now we pull the num_agents dimension into the batch
        playing = self.combine_batch_and_num_agents_dim(playing)
        # shape (batch * num_agents, num_agents, 1)

        ### OWN ID ###
        own_id_obs = orig_obs['own_id']
        # shape (batch, num_agents)
        own_id = self.combine_batch_and_num_agents_dim(own_id_obs)
        # shape (batch * num_agents)

        ### TRAITORS ###
        traitors_obs = None
        if self.traitor:
            traitors_obs = tf.convert_to_tensor(orig_obs['traitors'])
            # shape (num_agents, batch, 2)
            traitors_obs = tf.transpose(traitors_obs, [1,0,2])
            # Last dim is one-hot-encoded again. Only use second value
            traitors_obs = traitors_obs[:,:,1]
            # shape (batch, num_agents)
            traitors = self.combine_batch_and_num_agents_dim(traitors_obs)
            # shape (batch * num_agents)

        ### Main model
        main_net_ins = [
            map_and_agents, 
            votes,
            playing,
            own_id] \
            + ([traitors] if self.traitor else [])

        general_out, votes_logits = self.main_model(main_net_ins)
        # Split batch dim again
        votes_logits = self.split_batch_and_num_agents_dim(votes_logits)

        ### Move and Value model
        # Split batch dim again
        general_out = self.split_batch_and_num_agents_dim(general_out)
        # shape (batch, num_agents, net_outputs)
        move_logits, values = self.move_value_model(general_out)

        self._value_out = tf.reshape(values, [-1])
        return tf.concat([move_logits, tf.reshape(votes_logits, [-1, self.num_agents * 2])], axis=1), []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

    def combine_batch_and_num_agents_dim(self, tensor):
        # Takes tensor with shape (batch, num_agents, *other_dims)
        # and reshapes it to (batch * num_agents, *other_dims)
        last_dims = tensor.shape.as_list()[2:]
        return tf.reshape(tensor, shape=[-1] + last_dims)
        
    def split_batch_and_num_agents_dim(self, tensor):
        # Takes tensor with shape (batch * num_agents, *other_dims)
        # and reshapes it to (batch, num_agents, *other_dims)
        last_dims = tensor.shape.as_list()[1:]
        return tf.reshape(tensor, shape=[-1, self.num_agents] + last_dims)
