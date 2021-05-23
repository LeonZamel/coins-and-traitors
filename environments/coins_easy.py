import numpy as np

from gridworld_env import GridworldEnv

from util import CHAR_TO_COLOR

REWARD_EACH_TIMESTEP = -0.01
REWARD_COIN = 10
DEFAULT_NUM_COINS = 1
COIN_CHAR = 'A'
OTHER_AGENTS_CHAR = '1'


class CoinsEasy(GridworldEnv):
    """
    In this environment, there is only one type of agent, but multiple agents of this type can be spawned. 
    They must collect a number of coins as quickly as possible, they all get a reward if a coin is collected by one of them.
    There is only a limited view range.

    Expected learned behaviour:
    The agents split up and collect the coins as fast as possible

    """
    def __init__(self, env_config):
        super().__init__(env_config)

        # Grab config parameters
        self.num_coins = env_config.get('num_coins', DEFAULT_NUM_COINS)
        self.collected_coins = 0

        self.reset()

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        # Do actions
        actions = {}
        for agent_id, action in action_dict.items():
            # We must map the id of the action which we get from the policy to the action it corresponds to.
            # I.e. assign meaning to the id
            agent_action = self.agents[agent_id].movement_action_map[action]
            actions[agent_id] = agent_action
        self.move_agents(actions)

        # Get new observations
        obs = self.get_agents_observations()

        # Get new rewards
        # Reward if any agent reaches a goal
        # Default reward for each timestep
        reward = REWARD_EACH_TIMESTEP
        for agent in self.agents.values():
            if self.map[tuple(agent.pos)] == COIN_CHAR:
                self.map[tuple(agent.pos)] = ' '
                self.collected_coins += 1
                reward += REWARD_COIN

        # Give same reward to all agents
        rew = {id: reward for id in self.agents}

        # Check if all coins are collected
        done = {id: self.collected_coins == self.num_coins for id in self.agents}
        done["__all__"] = self.collected_coins == self.num_coins

        return obs, rew, done, info

    def reset(self):
        super().reset()
        self.collected_coins = 0

        # Reset coins
        coins_positions = self.map.sample_random_coordinates(self.num_coins, False)
        for p in coins_positions:
            self.map[tuple(p)] = COIN_CHAR

        agents_positions = self.map.sample_random_coordinates(self.num_agents, False)
        for i, agent in enumerate(self.agents.values()):
            agent.pos = agents_positions[i]

        return self.get_agents_observations()


    def get_agents_observations(self):
        map_with_agents = np.copy(self.map.data)

        # Put other agents into view
        for agent in self.agents.values():
            map_with_agents[tuple(agent.pos)] = OTHER_AGENTS_CHAR

        views = {
            id: self.map.get_view(map_with_agents, agent.pos, self.view_range, '0') for id, agent in self.agents.items()
        }

        for id, view in views.items():
            views[id] = self.map.encode(view)

        return views
