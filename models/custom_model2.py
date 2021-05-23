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


class CustomComplexInputNetwork2(TFModelV2):
    """
    Custom model for the coins traitors voting environment.
    Inputs are:
    - a general map view with walls, coins, and broken coins
    - a mapo view containing the observable agents
    - the current votes as a nxn matrix
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

        map_and_agents_input = [tf.keras.layers.Input(shape=(map_height, map_width, map_channels+1), 
            name=f'map_and_agents_observations_{i}') for i in range(self.num_agents)]
        votes_input = tf.keras.layers.Input(shape=(self.num_agents, self.num_agents, 1), name='votes_observations')

        playing_input = tf.keras.layers.Input(shape=(self.num_agents, 1), name='playing_observations')
        own_id_input = tf.keras.layers.Input(shape=(self.num_agents, ), name='own_id_observation')

        traitors_input = None
        if self.traitor:
            # A one hot encoding of who the traitors are. Only the traitors can see this
            traitors_input = tf.keras.layers.Input(shape=(self.num_agents, ), name='traitors_observation')

        # own_id and traitors slicing
        own_id_slices = []
        traitors_slices = []
        for i in range(self.num_agents):
            own_id_slices.append(tf.keras.layers.Lambda(lambda x: x[:,i:i+1])(own_id_input))
            if self.traitor:
                traitors_slices.append(tf.keras.layers.Lambda(lambda x: x[:,i:i+1])(traitors_input))

        # Map and agents observations net
        layer0 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')
        layer1 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')
        layer2 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')
        layer3 = tf.keras.layers.Flatten()
        layer4 = tf.keras.layers.Dense(
            128,
            activation='relu')

        playing_avg = tf.keras.layers.GlobalAvgPool1D()(playing_input)
        playing_avg = tf.keras.layers.Flatten()(playing_avg)

        map_and_agents_layers = []

        for i, m_a_input in enumerate(map_and_agents_input):
            last = layer0(m_a_input)
            last = layer1(last)
            last = layer2(last)
            last = layer3(last)
            last = tf.keras.layers.Concatenate(axis=1)([last, own_id_slices[i], playing_avg] + (traitors_slices if self.traitor else []))
            last = layer4(last)
            map_and_agents_layers.append(last)

        map_and_agents_add = tf.keras.layers.Add()(map_and_agents_layers)
        #map_and_agents_concat = tf.keras.layers.Concatenate(axis=2)(map_and_agents_layers)
        #map_and_agents_max = tf.keras.layers.MaxPool3D(
            #pool_size=(1,1,self.num_agents * 32))(map_and_agents_concat)


        # Votes observations net
        reshape_layer = votes_input

        last_layer_h_max = tf.keras.layers.MaxPooling2D(
            pool_size=(1,3),
            data_format='channels_last')(reshape_layer)
        last_layer_h_max = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_h_max)
        
        last_layer_h_avg = tf.keras.layers.AveragePooling2D(
            pool_size=(1,3),
            data_format='channels_last')(reshape_layer)
        last_layer_h_avg = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_h_avg)

        last_layer_v_max = tf.keras.layers.MaxPooling2D(
            pool_size=(3,1),
            data_format='channels_last')(reshape_layer)
        last_layer_v_max = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_v_max)
        
        last_layer_v_avg = tf.keras.layers.AveragePooling2D(
            pool_size=(3,1),
            data_format='channels_last')(reshape_layer)
        last_layer_v_avg = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_v_avg)

        votes_and_map_and_agents_layers = []

        votes_added = tf.keras.layers.Add()([last_layer_h_max, last_layer_h_avg, last_layer_v_max, last_layer_h_avg])
        votes_added = tf.keras.layers.Flatten()(votes_added)

        fc_layer0 = tf.keras.layers.Dense(
            128,
            activation='relu')
        fc_layer1 = tf.keras.layers.Dense(
            2,
            activation='relu')


        for i, m_a_l in enumerate(map_and_agents_layers):
            l_h_max = tf.keras.layers.Lambda(lambda x: x[:,i,:,:])(last_layer_h_max)
            l_h_max = tf.keras.layers.Flatten()(l_h_max)
            l_h_avg = tf.keras.layers.Lambda(lambda x: x[:,i,:,:])(last_layer_h_avg)
            l_h_avg = tf.keras.layers.Flatten()(l_h_avg)

            l_v_max = tf.keras.layers.Lambda(lambda x: x[:,:,i,:])(last_layer_v_max)
            l_v_max = tf.keras.layers.Flatten()(l_v_max)
            l_v_avg = tf.keras.layers.Lambda(lambda x: x[:,:,i,:])(last_layer_v_avg)
            l_v_avg = tf.keras.layers.Flatten()(l_v_avg)

            l = tf.keras.layers.Concatenate(axis=1)([own_id_slices[i], playing_avg, m_a_l, l_h_max, l_h_avg, l_v_max, l_v_avg, votes_added] + (traitors_slices if self.traitor else []))
            l = fc_layer0(l)
            l = fc_layer1(l)
            votes_and_map_and_agents_layers.append(l)

        move_logits = tf.keras.layers.Dense(
            128,
            activation='relu')(map_and_agents_add)
        move_logits = tf.keras.layers.Dense(
            128,
            activation='relu')(move_logits)
        move_logits = tf.keras.layers.Dense(
            5,
            activation='relu')(move_logits)

        logits = tf.keras.layers.Concatenate(axis=1)([move_logits] + votes_and_map_and_agents_layers)

        self.logits_and_value_model = None
        self._value_out = None


        # Create the value branch model.
        l = tf.keras.layers.Concatenate()([map_and_agents_add, votes_added])
        value_layer = tf.keras.layers.Dense(
            1,
            name='value_out',
            activation=None,
            kernel_initializer=normc_initializer(0.01))(l)
        self.logits_and_value_model = tf.keras.models.Model(
            map_and_agents_input + [votes_input, playing_input, own_id_input] + ([traitors_input] if self.traitor else []), [logits, value_layer])


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and 'obs_flat' in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS],
                                                   self.obs_space, 'tf')

        # Process inputs

        # We get a 2D Python array like [[<Tensor, shape (?,2)>, ...], ...], where '?' is the batch size
        # so we need to first convert this to a tensor of shape (num_agent, num_agents, ?, 2)
        # and then pull out the '?' dimension to the front so we get (num_agent, num_agents, 2) tensors as votes 
        votes_obs = tf.transpose(tf.convert_to_tensor(orig_obs['votes']), [2,0,1,3])
        # The votes are one hot encoded, so (1,0) means no vote and (0,1) means vote
        # We can just take the second component and collapse the dimension down to size 1
        votes_obs = votes_obs[:,:,:,1]

        # The map has shape (batchsize, viewsize, viewsize, channels)
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

        playing_obs = tf.convert_to_tensor(orig_obs['playing'])
        playing_obs = tf.transpose(playing_obs, [1,0,2])
        # Similar to the votes above this is one hot encoded. However, we instead slice form 1:2 so the dimension of size 1 stays
        # we keep it for the 1D convolution above
        playing_obs = playing_obs[:,:,1:2] 

        own_id_obs = orig_obs['own_id']

        traitors_obs = None
        if self.traitor:
            traitors_obs = tf.convert_to_tensor(orig_obs['traitors'])
            traitors_obs = tf.transpose(traitors_obs, [1,0,2])
            traitors_obs = traitors_obs[:,:,1]

        ins = map_and_agents_obs + [votes_obs, playing_obs, own_id_obs] + ([traitors_obs] if self.traitor else [])

        # Logits- and value outputs
        logits, values = self.logits_and_value_model(ins)
        self._value_out = tf.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

