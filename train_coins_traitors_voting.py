import os

import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
from ray.rllib.agents import Trainer
from ray.rllib.env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog

import cv2

from environments.coins_traitors_voting import CoinsTraitorsVoting
from models.custom_model import CustomComplexInputNetwork
from models.custom_model2 import CustomComplexInputNetwork2
from models.custom_model3 import CustomComplexInputNetwork3
from models.custom_model3_pytorch import CustomComplexInputNetwork3Torch
from models.perm_equivariant import PermEquivariantModel

###### ENV CONFIG ######
env_config = {
    'num_agents': 3,
    'num_traitors': 1,
    'num_coins': 10,
    'view_ranges': {'traitors': 3, 'innocents': 3},
    'agent_removal': 'voting', # Can be 'voting' or 'attacking'
    'observable_votes': True, # If agents can see the votes
    'horizon': 100, # After how many timesteps the episode is ended
    'sticky_votes': 100, # How long a True vote should stay stuck at True
    'vote_majority_only_alive': True, # How to determine the majority for voting. Only take into account alive agents or also removed agents
    'reset_removed_agents_votes': True, # If the votes of a removed agent should be reset to all False (not taken into acount anymore) or should be kept
    'disable_self_votes': True, # If agents should not be able to vote for themselves
    'rewards': {
        'innocents': {
            'per_timestep': -0.1,
            'collect_coin': 10,
            'break_coin': -10,
            'remove_innocent': -60,
            'remove_traitor': 10,
        },
        'traitors': {
            'per_timestep': 0,
            'collect_coin': -10,
            'break_coin': 10,
            'remove_innocent': 10,
            'remove_traitor': -60,
        }
    },

    'debug': {
        'disable_votes': False,
        'verbose': False,
    }
}
#######################


results_path = os.path.join(os.getcwd(), 'data')

model_name = 'custom_vision'
ModelCatalog.register_custom_model(model_name, CustomComplexInputNetwork)

model_name2 = 'custom_vision2'
ModelCatalog.register_custom_model(model_name2, CustomComplexInputNetwork2)

model_name3 = 'custom_vision3'
ModelCatalog.register_custom_model(model_name3, CustomComplexInputNetwork3)

model_name3_torch = 'custom_vision3_torch'
ModelCatalog.register_custom_model(model_name3_torch, CustomComplexInputNetwork3Torch)

model_perm_equivariant = 'model_perm_equivariant'
ModelCatalog.register_custom_model(model_perm_equivariant, PermEquivariantModel)

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
        # We want to supply some basic info/stats. In RLlib multiagent envs the info dict cannot contain keys which are not agent IDs.
        # For this reason we just pick one agent that is still in the game and pass the info through their ID. This must be "untangled" here
        ep_info = None
        for i in range(env_config['num_agents']):
            inf = episode.last_info_for(i)
            if inf is not None and inf:
                ep_info = episode.last_info_for(i)
                break
        if ep_info is not None and ep_info:
            # The rewards are sometimes reported multiple times and not just in the episode that they occured
            # Therefore, these first two metrics are a bit buggy and should not be used for deep analysis
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

# We create a dummy environment which gives us the correct spaces for our config
_temp_env = CoinsTraitorsVoting(env_config)
_observation_spaces = _temp_env.get_observation_spaces()
_action_spaces = _temp_env.get_action_spaces()

config = {
    'env': CoinsTraitorsVoting,
    'horizon': env_config['horizon'],
    'callbacks': EvaluationCallbacks,
    'env_config': env_config,
    'multiagent': {
        'policies': {
            # the first tuple value is None -> uses default policy
            'innocent': (None, _observation_spaces['innocents'], _action_spaces['innocents'], {
                'model': {
                    'custom_model': model_perm_equivariant,
                    'custom_model_config': {
                        'traitor': False,
                    },
                    # We need to specify the convolutional filters for the network, as these cannot be inferred automatically
                    # out_channels, kernel, stride
                    #'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2]],
                    #'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2], [128, [3, 3], 2]]
                }
            }),
            'traitor': (None, _observation_spaces['traitors'], _action_spaces['traitors'], {
                'model': {
                    'custom_model': model_perm_equivariant,
                    'custom_model_config': {
                        'traitor': True,
                    },
                    # We need to specify the convolutional filters for the network, as these cannot be inferred automatically
                    # out_channels, kernel, stride
                    #'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2]]
                    #'conv_filters': [[8, [3, 3], 2], [32, [4, 4], 2], [128, [3, 3], 2]]
                }
            }),
        },
        'policy_mapping_fn':
            lambda agent_id: 'traitor' if agent_id < env_config['num_traitors'] else 'innocent',
    },
    
    'num_workers': 1,
    # Sometimes slower using GPU. Probably too much overhead if the network is simple
    # 'num_gpus': 1,
    'framework': 'torch',
}

def train():
    stop = {
        'training_iteration': 5000,
        'timesteps_total': 10000000,
    }
    results = tune.run(trainer_type, config=config, stop=stop, checkpoint_freq=25, checkpoint_at_end=True, local_dir=results_path)

def test(render_type='print'):
    # render_type can either be 'print' or 'image'
    trainer = trainer_type(env=CoinsTraitorsVoting, config=config)
    # Change this path when testing
    checkpoint_path = os.path.join(results_path, '3-agents-no-self-votes-sticky-100/1/checkpoint_1150/checkpoint-1150')
    trainer.restore(checkpoint_path)
    env = CoinsTraitorsVoting(env_config=env_config)
    imgs = run_episode(trainer, env, env_config['horizon'], True, render_type)
    if render_type == 'image':
        height, width, channels = imgs[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('video.avi',fourcc,1,(width,height))
        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()

def run_episode(trainer: Trainer, env: MultiAgentEnv, horizon, render=True, render_type='print'):
    episode_rewards = {agent: 0 for agent in env.agents}
    done = {agent: False for agent in env.agents}
    done['__all__'] = False
    obs = env.reset()
    imgs = []
    if render:
        imgs.append(env.render(render_type))
    steps = 0
    while not done['__all__']:
        actions = {agent_id: trainer.compute_action(obs, policy_id=config['multiagent']['policy_mapping_fn'](agent_id)) for agent_id, obs in obs.items() if not done[agent_id]}
        obs, rew, done, info = env.step(actions)
        for agent, reward in rew.items():
            episode_rewards[agent] += reward
        steps += 1
        if render:
            imgs.append(env.render(render_type))
        if steps == horizon:
            done['__all__'] = True

    print('Rewards by agent id: ', episode_rewards)
    print('Number of steps: ', steps)

    return imgs


if __name__ == '__main__':
    ray.init()
    # If you want to train a policy, call train() here. If you want to test/visualize it, call test() or test('image') 
    test('image')
    ray.shutdown()
