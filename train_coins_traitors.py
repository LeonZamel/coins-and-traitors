import os

import numpy as np

import gym, ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import Trainer
from ray.rllib.env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks

from environments.coins_traitors import CoinsTraitors

results_path = os.getcwd() + '/run_results/'

VIEW_RANGE = 3
EXTENDED_VIEW_RANGE = 5

DEFAULT_ACTION_SPACE = gym.spaces.Discrete(5)
DEFAULT_OBSERVATION_SPACE = gym.spaces.Box(low=0.0, high=255.0, shape=(2 * VIEW_RANGE + 1,
                                                2 * VIEW_RANGE + 1, 3), dtype=np.float32)

ACTIONS_WITH_ATTACK = gym.spaces.Discrete(6)
EXTENDED_OBSERVATION_SPACE = gym.spaces.Box(low=0.0, high=255.0, shape=(2 * EXTENDED_VIEW_RANGE + 1,
                                                2 * EXTENDED_VIEW_RANGE + 1, 3), dtype=np.float32)

num_agents = 3
num_traitors = 1



trainer_type = ppo.PPOTrainer

class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data['reward_innocents'] = [0]
        episode.user_data['reward_traitors'] = [0]
        episode.user_data['coins_collected'] = 0
        episode.user_data['coins_destroyed'] = 0
        episode.user_data['removed_innocents'] = 0
        episode.user_data['removed_traitors'] = 0

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        ep_info = None
        for i in range(num_agents):
            inf = episode.last_info_for(i)
            if inf is not None and inf:
                ep_info = episode.last_info_for(i)
                break
        if ep_info is not None and ep_info:
            episode.user_data['reward_innocents'].append(ep_info['reward']['innocents'])
            episode.user_data['reward_traitors'].append(ep_info['reward']['traitors'])
            episode.user_data['coins_collected'] = ep_info['coins']['collected']
            episode.user_data['coins_destroyed'] = ep_info['coins']['destroyed']
            episode.user_data['removed_innocents'] = ep_info['removed']['innocents']
            episode.user_data['removed_traitors'] = ep_info['removed']['traitors']


    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics['reward_innocents'] = sum(episode.user_data['reward_innocents'])
        episode.custom_metrics['reward_traitors'] = sum(episode.user_data['reward_traitors'])
        episode.custom_metrics['coins_collected'] = episode.user_data['coins_collected']
        episode.custom_metrics['coins_destroyed'] = episode.user_data['coins_destroyed']
        episode.custom_metrics['removed_innocents'] = episode.user_data['removed_innocents']
        episode.custom_metrics['removed_traitors'] = episode.user_data['removed_traitors']

env_config = {
    'num_agents': num_agents,
    'num_traitors': num_traitors,
    'num_coins': 10,
    'attack_range': 2,
    'view_ranges': {'traitors': 3, 'innocents': 3},
    'observable_id': True,
}

config = {
    'env': CoinsTraitors,
    'horizon': 200,
    'callbacks': EvaluationCallbacks,
    'env_config': env_config,
    'multiagent': {
        'policies': {
            # the first tuple value is None -> uses default policy
            'innocent': (None, DEFAULT_OBSERVATION_SPACE, ACTIONS_WITH_ATTACK, {
                # We need to specify the convolutional filters for the network, as these cannot be inferred automatically
                'model': {
                    # out_channels, kernel, stride
                    'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2]],
                    #'use_lstm': True,
                }
            }),
            'traitor': (None, EXTENDED_OBSERVATION_SPACE, DEFAULT_ACTION_SPACE, {
                # We need to specify the convolutional filters for the network, as these cannot be inferred automatically
                'model': {
                    # out_channels, kernel, stride
                    'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2], [128, [3, 3], 2]]
                }
            }),
        },
        'policy_mapping_fn':
            lambda agent_id: 'traitor' if agent_id < num_traitors else 'innocent',

    },
    
    'num_workers': 2,
    # Slower using GPU. Probably too much overhead for such a simple network
    # 'num_gpus': 0.2,
    # 'framework': 'torch',
}

def train():
    stop = {
        'training_iteration': 1000,
        'timesteps_total': 3000000,
        # 'episode_reward_mean': 89.5,
    }
    results = tune.run(trainer_type, config=config, stop=stop, checkpoint_freq=10, checkpoint_at_end=True, local_dir=results_path)

def test():
    trainer = trainer_type(env=CoinsTraitors, config=config)
    checkpoint_path = results_path + 'PPO_2021-03-11_13-50-28\PPO_CoinsTraitors_5b1df_00000_0_2021-03-11_13-50-28\checkpoint_430\checkpoint-430'
    trainer.restore(checkpoint_path)
    env = CoinsTraitors(env_config=env_config)
    run_episode(trainer, env, True)

def run_episode(trainer: Trainer, env: MultiAgentEnv, render=True):
    episode_rewards = {agent: 0 for agent in env.agents}
    done = {agent: False for agent in env.agents}
    done['__all__'] = False
    obs = env.reset()

    steps = 0
    while not done['__all__']:
        actions = {agent_id: trainer.compute_action(obs, policy_id=config['multiagent']['policy_mapping_fn'](agent_id)) for agent_id, obs in obs.items() if not done[agent_id]}
        obs, rew, done, info = env.step(actions)
        for agent, reward in rew.items():
            episode_rewards[agent] += reward
        steps += 1
        if render:
            env.render()

    print('Rewards by agent id: ', episode_rewards)
    print('Number of steps: ', steps)


if __name__ == '__main__':
    ray.init()
    train()
    ray.shutdown()
