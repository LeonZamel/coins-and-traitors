import numpy as np

import gym, ray
from ray import tune

from gridworld_env import GridworldEnv
from environments.coins_easy import CoinsEasy


VIEW_RANGE = 3
DEFAULT_ACTION_SPACE = gym.spaces.Discrete(5)
DEFAULT_OBSERVATION_SPACE = gym.spaces.Box(low=0.0, high=255.0, shape=(2 * VIEW_RANGE + 1,
                                                2 * VIEW_RANGE + 1, 3), dtype=np.float32)

if __name__ == "__main__":
    ray.init()

    config={
        "env": CoinsEasy,
        "horizon": 200,
        "env_config": {
            "num_agents": 3,
            "num_coins": 3,
            "view_range": VIEW_RANGE,
            'layer_map': {'A': 0, 'C': 1}
        },
        "multiagent": {
            "policies": {
                # the first tuple value is None -> uses default policy
                "default": (None, DEFAULT_OBSERVATION_SPACE, DEFAULT_ACTION_SPACE, {}),
            },
            "policy_mapping_fn":
                lambda agent_id:
                    "default",

        },
        # We need to specify the convolutional filters for the network, as these cannot be inferred automatically
        "model": {
            # out_channels, kernel, stride
            "conv_filters": [[8, [3, 3], 2], [32, [4, 4], 2]]
        },

        "num_workers": 4,
        # Slower using GPU. Probably too much overhead for such a simple network
        # "num_gpus": 0.2,
        # "framework": "torch",
    }

    stop = {
        "training_iteration": 1000,
        "timesteps_total": 2000000,
        "episode_reward_mean": 89.5,
    }

    results = tune.run("PPO", config=config, stop=stop)

    ray.shutdown()