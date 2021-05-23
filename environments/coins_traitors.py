import numpy as np
import pandas as pd

from gridworld_agent import GridworldAgent, MOVEMENT_ACTIONS
from gridworld_env import GridworldEnv

COIN_CHAR = 'A'
BROKEN_COIN_CHAR = 'B'
BROKEN_COIN_CHAR_DECAYED = 'C'
BROKEN_COIN_DECAY_STEPS = 3

OTHER_AGENTS_INNOCENT_CHAR = '1'
OTHER_AGENTS_TRAITOR_CHAR = '2'


class CoinsTraitors(GridworldEnv):
    """
    In this environment, there are two types of agents. There are good/innocent and bad/traitor/adversary agents.
    Innocents cannot see the type of other agents, traitors can.
    The good agents want to collect the coins, as each of them gets a reward for each coin collected. 
    The traitor(s) get a negative reward for each coin collected, however, they can break coins so that they can no longer be collected by innocents.
    Innocents can attack other agents. Once an agent is attacked, they are removed for that round. If an innocent removes another innocent,
    they get a negative reward. If they remove a traitor, they get a positive reward.
    There is a limited view range.

    Expected learned behaviour:
    The innocents try to collect the coins as fast as possible. The traitors try to destroy coins without being seen. If an innocent sees a traitor
    breaking a coin, they attack the traitor.
    """

    def __init__(self, env_config):
        # Grab config parameters
        self.num_coins = env_config['num_coins']
        self.num_traitors = env_config['num_traitors']
        self.attack_range = env_config['attack_range']
        self.observable_id = env_config['observable_id']

        super().__init__(env_config)

        assert self.num_traitors < self.num_agents, 'There must be less adversaries than agents total'

        self.collected_coins = 0
        self.destroyed_coins = 0
        self.removed_agents = {}
        self.coins = []

        # Which agent id is mapped to which observable id, e.g. {0: 1} means the agent 0 is visible to others as 1
        self.observable_id_assignment = {}

        self.reset()

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        # Default rewards each timestep
        reward_innocents = self.rewards['innocents']['per_timestep']
        reward_traitors = self.rewards['traitors']['per_timestep']

        # Do actions
        actions = {}
        for agent_id, action in action_dict.items():
            # We must map the id of the action which we get from the policy to the action it corresponds to.
            # I.e. assign meaning to the id
            agent_action = self.agents[agent_id].movement_action_map[action]
            actions[agent_id] = agent_action
        
        # Determine which agents are attacked
        to_remove_ids = set()
        for id, action in actions.items():
            if action == MOVEMENT_ACTIONS.ATTACK:
                agent = self.agents[id]
                for id2, agent2 in self.agents.items():
                    if id != id2:
                        if (np.abs(agent.pos - agent2.pos) <= self.attack_range).all():
                            # Remember to remove the attacked agent
                            to_remove_ids.add(id2)

        # Actually remove the attacked agents
        # We can't do this above as the agents are iterated in that loop
        for id in to_remove_ids:
            a_type = "traitor" if self.agents[id].traitor else "innocent"
            if self.agents[id].traitor:
                reward_innocents += self.rewards['innocents']['remove_traitor']
                reward_traitors += self.rewards['traitors']['remove_traitor']
            else:
                reward_innocents += self.rewards['innocents']['remove_innocent']
                reward_traitors += self.rewards['traitors']['remove_innocent']

            # print(f"Removed {id} which was a {a_type}")
            self.removed_agents[id] = self.agents[id]
            del self.agents[id]

        # Filter actions for removed agents
        actions = {id: action for id, action in actions.items() if id not in self.removed_agents.keys()}
        
        self.move_agents(actions)

        # Update broken coins
        for coin in self.coins:
            if coin.broken:
                coin.decay_one_step()

        # Interact with coins
        for agent in self.agents.values():
            new_coins = []
            for coin in self.coins:
                collected = False
                if tuple(agent.pos) == coin.pos and not coin.broken:
                    if agent.traitor:
                        coin.broken = True
                        self.destroyed_coins += 1
                        reward_innocents += self.rewards['innocents']['break_coin']
                        reward_traitors += self.rewards['traitors']['break_coin']
                        # print("Broke coin")
                    else:
                        collected = True
                        self.collected_coins += 1
                        reward_innocents += self.rewards['innocents']['collect_coin']
                        reward_traitors += self.rewards['traitors']['collect_coin']
                        # print("Collected coin")
                if not collected:
                    new_coins.append(coin)
            self.coins = new_coins

        # Get new observations
        obs = self.get_agents_observations()

        # Give same reward to agents by type
        rew = {id: reward_traitors if agent.traitor else reward_innocents for id, agent in self.agents.items()}

        done = {id: False for id in self.agents}
        done2 = {id: True for id in self.removed_agents}
        done.update(done2)
        # Check if all coins are either collected or destroyed
        done["__all__"] = self.collected_coins + self.destroyed_coins == self.num_coins

        if not list(self.agents.keys()):
            info = {}
        else:
            info = {list(self.agents.keys())[0]: {
                'reward': {
                    'innocents': reward_innocents,
                    'traitors': reward_traitors,
                },
                'coins': {
                    'collected': self.collected_coins,
                    'destroyed': self.destroyed_coins,
                },
                'removed': {
                    'innocents': len(list(filter(lambda a: not a.traitor, self.removed_agents.values()))),
                    'traitors': len(list(filter(lambda a: a.traitor, self.removed_agents.values()))),
                }
            }}

        return obs, rew, done, info

    def reset(self):
        super().reset()
        self.collected_coins = 0
        self.destroyed_coins = 0
        self.agents.update(self.removed_agents)
        self.removed_agents = {}
        self.broken_coins_age = {}

        if self.observable_id:
            perm = np.random.permutation(self.num_agents)
            self.observable_id_assignment = {id: p for id, p in zip(self.agents.keys(), perm)}

        # Reset coins
        coins_positions = self.map.sample_random_coordinates(self.num_coins, False)
        for p in coins_positions:
            self.map[tuple(p)] = COIN_CHAR

        agents_positions = self.map.sample_random_coordinates(self.num_agents, False)
        for i, agent in enumerate(self.agents.values()):
            agent.pos = agents_positions[i]

        return self.get_agents_observations()


    def get_agents_observations(self):
        if self.observable_id:
            # Agents see an id for every other agent
            map_with_agents = np.copy(self.map.data)
            for id, agent in self.agents.items():
                map_with_agents[tuple(agent.pos)] = str(self.observable_id_assignment[id])

            # Slice the view for each agent
            views = {
                id: self.map.get_view(map_with_agents, agent.pos, agent.view_range, '0') 
                for id, agent in self.agents.items()
            }

        else:
            innocent_map_with_agents = np.copy(self.map.data)
            traitor_map_with_agents = np.copy(self.map.data)

            # Put other agents into view
            # We create two maps. One for the traitors and one for the innocents, since only the traitors can see the type
            for agent in self.agents.values():
                innocent_map_with_agents[tuple(agent.pos)] = OTHER_AGENTS_INNOCENT_CHAR
                traitor_map_with_agents[tuple(agent.pos)] = OTHER_AGENTS_TRAITOR_CHAR if agent.traitor else OTHER_AGENTS_INNOCENT_CHAR

            # Slice the view for each agent
            # We assign the corresponding views to the agents by type
            views = {
                id: self.map.get_view(traitor_map_with_agents if agent.traitor else innocent_map_with_agents, agent.pos, agent.view_range, '0') 
                for id, agent in self.agents.items()
            }

        for id, view in views.items():
            views[id] = self.map.encode(view)

        return views

    def init_agents(self):
        """
        Create the agent objects
        """
        for i in range(self.num_agents):
            self.agents[i] = GridworldAgent(
                i, (0, i), self.view_range[i], self.map, i < self.num_traitors
            )
    
    def render(self):
        traitor_map_with_agents = np.copy(self.map.data)

        for agent in self.agents.values():
            traitor_map_with_agents[tuple(agent.pos)] = OTHER_AGENTS_TRAITOR_CHAR if agent.traitor else OTHER_AGENTS_INNOCENT_CHAR
        
        print(pd.DataFrame(traitor_map_with_agents))
