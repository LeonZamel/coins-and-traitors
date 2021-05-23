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

tf1, tf, tfv = try_import_tf()


class CustomComplexInputNetwork(TFModelV2):
    """
    Model for using multiple 2D inputs
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        
        self.original_space = obs_space.original_space

        map_and_agents_net = None
        votes_net = None

        concat_size = 0

        self.num_agents = len(self.original_space['votes'])
        map_height, map_width, map_channels = self.original_space['map'].shape

        map_and_agents_input = [tf.keras.layers.Input(shape=(map_height, map_width, map_channels+1), 
            name=f'map_and_agents_observations_{i}') for i in range(self.num_agents)]
        votes_input = tf.keras.layers.Input(shape=(self.num_agents, self.num_agents, 1), name='votes_observations')

        # Map / agents observations net
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
        layer2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        layer3 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3,3),
            data_format='channels_last', 
            padding='same',
            activation='relu')
        layer4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        layer5 = tf.keras.layers.Flatten()

        map_and_agents_net = []

        for m_a_input in map_and_agents_input:
            last = layer0(m_a_input)
            last = layer1(last)
            last = layer2(last)
            last = layer3(last)
            last = layer4(last)
            last = layer5(last)
            map_and_agents_net.append(last)

        map_and_agents_net = tf.keras.layers.Add()(map_and_agents_net)

        # Votes observations net
        reshape_layer = votes_input
        last_layer_h = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(reshape_layer)
        last_layer_h = tf.keras.layers.MaxPooling2D(
            pool_size=(1,3),
            data_format='channels_last')(last_layer_h)
        last_layer_h = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_h)

        last_layer_v = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(reshape_layer)
        last_layer_v = tf.keras.layers.MaxPooling2D(
            pool_size=(3,1),
            data_format='channels_last')(last_layer_v)
        last_layer_v = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer_v)

        last_layer = tf.keras.layers.Add()([last_layer_h, last_layer_v])
        last_layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1,1),
            data_format='channels_last', 
            padding='valid',
            activation='relu')(last_layer)
        votes_net = tf.keras.layers.Flatten()(last_layer)

        # Combining nets
        concat_size += map_and_agents_net.shape[1]
        concat_size += votes_net.shape[1]
       
        self.logits_and_value_model = None
        self._value_out = None

        # Concatenate and create fully connected layers
        concat_layer = tf.keras.layers.Concatenate(axis=1)([map_and_agents_net, votes_net])

        last_layer = tf.keras.layers.Dense(
            128,
            activation='relu')(concat_layer)
        last_layer = tf.keras.layers.Dense(
            128,
            activation='relu')(last_layer)

        # Action-distribution head.
        logits_layer = tf.keras.layers.Dense(
            num_outputs,
            activation=tf.keras.activations.linear,
            name='logits')(last_layer)

        # Create the value branch model.
        value_layer = tf.keras.layers.Dense(
            1,
            name='value_out',
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
        self.logits_and_value_model = tf.keras.models.Model(
            map_and_agents_input + [votes_input], [logits_layer, value_layer])


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
        agents_obs = tf.convert_to_tensor(orig_obs['agents'])
        # We first transpose to (num_agents+1, batchsize, viewsize, viewsize)
        agents_obs = tf.transpose(agents_obs, [3,2,0,1])

        # We can now concatenate each agent view with the map view
        map_and_agents_obs = []
        for i in range(self.num_agents):
            a_obs = tf.expand_dims(agents_obs[i], 3)
            map_and_agents_obs.append(tf.concat([map_obs, a_obs], axis=3))

        ins = map_and_agents_obs + [votes_obs]

        # Logits- and value branches.
        logits, values = self.logits_and_value_model(ins)
        self._value_out = tf.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

