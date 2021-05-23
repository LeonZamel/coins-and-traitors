import numpy as np
from numpy.core.numeric import zeros_like
import pandas as pd

import gym, ray

import cv2

from gridworld_agent import GridworldAgent, MOVEMENT_ACTIONS
from gridworld_env import GridworldEnv
from .coin import Coin

RENDERING_INNOCENT = 'I'
RENDERING_TRAITOR = 'T'
RENDERING_COIN = 'C'
RENDERING_BROKEN_COIN = 'B'

class CoinsTraitorsVoting(GridworldEnv):
    """
    In this environment, there are two types of agents. There are good/innocent and bad/traitor/adversary agents.
    Innocents cannot see the type of other agents, traitors can.
    Each agent is represented via an ID. This ID changes between episodes.
    The good agents want to collect the coins, as each of them gets a reward for each coin collected. 
    The traitor(s) get a negative reward for each coin collected, however, they can break coins so that they can no longer be collected by innocents.
    Broken coins are observable as such. They have a freshness value which should decrease via some function each timestep. It starts as 1 in the 
    round it gets destroyed and if it drops to 0 the broken coin is no longer observable.
    Agents can vote to remove other agents. For this, they vote for the IDs of the agents they want to remove. 
    If more than half of the (total or alive) agents vote for one agent, that agent is removed for that round. 
    If an innocent is removed, innocents get punished but traitors get a reward, if a traitor is removed, traitors get punished but innocents get a reward.
    There is a limited view range.

    Expected learned behaviour:
    The innocents try to collect the coins as fast as possible. The traitors try to destroy coins without being seen. If an innocent sees a traitor
    breaking a coin, they vote for the traitor. Traitors try to trick innocents into voting for other innocents. Traitors should not vote for traitors unless
    in very sophisticated strategies where traitors might try to gain trust of other innocents.
    """

    def __init__(self, env_config):
        # Grab config parameters
        self.num_coins = env_config['num_coins']
        self.num_traitors = env_config['num_traitors']
        self.observable_votes = env_config['observable_votes']
        self.agent_removal = env_config['agent_removal']
        self.sticky_votes = env_config['sticky_votes']
        self.vote_majority_only_alive = env_config['vote_majority_only_alive']
        self.reset_removed_agents_votes = env_config['reset_removed_agents_votes']
        self.disable_self_votes = env_config['disable_self_votes']
        self.view_ranges = env_config['view_ranges']
        self.innocents_view_range = self.view_ranges['innocents']
        self.traitors_view_range = self.view_ranges['traitors']
        
        # Helpful for debugging / testing
        self.debug_config = {
            'disable_votes': False, # Votes can still be seen and voting actions are still possible, however voting has no effect, only makes sense for debugging
            'verbose': False, # Enable some print outputs within the env 
        }
        self.debug_config.update(env_config.get('debug', {}))
        self.disable_votes = self.debug_config['disable_votes']
        self.verbose = self.debug_config['verbose']

        super().__init__(env_config)

        assert self.agent_removal in ['voting', 'attacking'], f'Unknown agent removal type "{self.agent_removal}"'
        assert self.agent_removal != 'attacking' or self.observable_votes, 'Votes cannot be observable when playing without votes'
        assert self.num_traitors < self.num_agents, 'There must be less traitors than agents total'

        self.collected_coins = 0
        self.destroyed_coins = 0
        self.removed_agents = {}
        self.coins = []

        # Which agent id is mapped to which observable id, e.g. {0: 1} means the agent 0 is visible to others as 1
        self.observable_id_assignment = {}

        # Who voted for whom, e.g. [[False, True, False], ...] means agent 0 voted for agent 1
        # This uses the actual IDs of the agents, not the IDs observable by agents
        self.votes = None
        # How long ago a vote was set to True the last time. We need this to reset votes after the 'sticky_votes' timesteps
        self.votes_time_ago = None

        # This is where we record the votes before possibly resetting them after an agent gets removed. 
        # We need this for the render function
        self._votes_pre_reset = None 
        

        self.reset()

    def step(self, action_dict):
        """
        One step of the environment. Most game logic lives here
        """
        obs, rew, done, info = {}, {}, {}, {}

        # Default rewards each timestep
        reward_innocents = self.rewards['innocents']['per_timestep']
        reward_traitors = self.rewards['traitors']['per_timestep']

        # Do actions
        move_actions = {}
        vote_actions = {}

        to_remove_ids = set()

        # Move actions
        for agent_id, action in action_dict.items():
            # We must map the id of the action which we get from the policy to the action it corresponds to.
            # I.e. assign meaning to the id
            move_actions[agent_id] = self.agents[agent_id].movement_action_map[action['move']]
            
        # Voting actions
        if self.agent_removal == 'voting' and not self.disable_votes:
            for agent_id, action in action_dict.items():
                # We must map back the vote for the observed id to the actual id
                inv_observable_id_assignment = {v: k for k, v in self.observable_id_assignment.items()}
                vote_vector_permutated = action['vote']
                vote_vector = np.copy(vote_vector_permutated)
                for i, v in enumerate(vote_vector_permutated):
                    vote_vector[inv_observable_id_assignment[i]] = v
                vote_actions[agent_id] = vote_vector

            # Update votes
            # First update timer
            self.votes_time_ago += 1
            # Reset old votes after the given number of timesteps
            self.votes[self.votes_time_ago > self.sticky_votes] = False
            # Add new votes
            for agent_id in vote_actions:
                for i, v in enumerate(vote_actions[agent_id]):
                    if v == 1:
                        self.votes[agent_id][i] = True
                        self.votes_time_ago[agent_id][i] = 0

            # If self votes are disabled, we reset those votes to False here
            if self.disable_self_votes:
                for agent_id in self.agents:
                    self.votes[agent_id][agent_id] = False
            
            # Determine to remove agents which were voted for by more than half the number of agents
            votes_for = np.zeros(self.num_agents, dtype=int)
            for id in self.agents:
                for other_id in self.agents:
                    if self.votes[id][other_id]:
                        votes_for[other_id] += 1
            
            for id, votes in enumerate(votes_for):
                if (self.vote_majority_only_alive and votes > len(self.agents) / 2) or \
                    (not self.vote_majority_only_alive and votes > self.num_agents / 2):
                    to_remove_ids.add(id)
        
        # Attacking actions
        elif self.agent_removal == 'attacking':
            for agent_id, action in move_actions.items():
                if action == MOVEMENT_ACTIONS.ATTACK:
                    agent = self.agents[agent_id]
                    for id2, agent2 in self.agents.items():
                        if agent_id != id2:
                            if (np.abs(agent.pos - agent2.pos) <= self.attack_range).all():
                                # Remember to remove the attacked agent
                                to_remove_ids.add(id2)

        # Calculate rewards for removing agents
        for id in to_remove_ids:
            a_type = "traitor" if self.agents[id].traitor else "innocent"
            if self.agents[id].traitor:
                reward_innocents += self.rewards['innocents']['remove_traitor']
                reward_traitors += self.rewards['traitors']['remove_traitor']
            else:
                reward_innocents += self.rewards['innocents']['remove_innocent']
                reward_traitors += self.rewards['traitors']['remove_innocent']

            if self.verbose:
                print(f"Removed {id} which was a {a_type}")

        # Move the agents
        self.move_agents(move_actions)

        # Update broken coins
        for coin in self.coins:
            if coin.broken:
                coin.decay_one_step()

        # Interact with coins
        for agent in self.agents.values():
            # new_coins will contain all coins which were not collected/broken
            new_coins = []
            for coin in self.coins:
                # We loop over every coin and check if it was collected or broken
                collected = False
                if tuple(agent.pos) == coin.pos and not coin.broken:
                    if agent.traitor:
                        coin.broken = True
                        self.destroyed_coins += 1
                        reward_innocents += self.rewards['innocents']['break_coin']
                        reward_traitors += self.rewards['traitors']['break_coin']
                        if self.verbose:
                            print("Broke coin")
                    else:
                        collected = True
                        self.collected_coins += 1
                        reward_innocents += self.rewards['innocents']['collect_coin']
                        reward_traitors += self.rewards['traitors']['collect_coin']
                        if self.verbose:
                            print("Collected coin")
                if not collected:
                    new_coins.append(coin)
            self.coins = new_coins

        # Get new observations
        obs = self.get_agents_observations()

        # Give same reward to agents by type
        rew = {id: reward_traitors if agent.traitor else reward_innocents for id, agent in self.agents.items()}

        # We want to supply some basic info/stats. In RLlib multiagent envs the info dict cannot contain keys which are not agent IDs.
        # For this reason we just pick one agent that is still in the game and pass the info through their ID. This must be "untangled" on the receiving side
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

        self._votes_pre_reset = np.copy(self.votes)

        # Actually remove the agents from the dicts, we only do this here at the end so we still receive
        # observations and rewards for the agents which get removed
        for id in to_remove_ids:
            if self.agent_removal == 'voting' and self.reset_removed_agents_votes:
                # We set the removed agents' votes to all False so they have no effect anymore
                self.votes[id] = np.zeros_like(self.votes[id])

            self.removed_agents[id] = self.agents[id]
            del self.agents[id]

        # Agents are done if they are removed from the game as they don't participate anymore for this round
        done = {id: False for id in self.agents}
        done2 = {id: True for id in self.removed_agents}
        done.update(done2)

        # Check if all coins are either collected or destroyed or all agents are removed
        # Then the environment ends by setting the __all__ attribute of done to True
        # Otherwise one episode ends after a fixed number of timesteps (horizon config variable)
        done["__all__"] = self.collected_coins + self.destroyed_coins == self.num_coins or len(list(self.removed_agents.values())) == self.num_agents

        return obs, rew, done, info

    def reset(self):
        """
        Reset the environment between episodes
        """
        super().reset()
        self.collected_coins = 0
        self.destroyed_coins = 0
        self.agents.update(self.removed_agents)
        self.removed_agents = {}
        self.coins = []

        self.votes = np.zeros((self.num_agents, self.num_agents), dtype=bool) if self.agent_removal == 'voting' else None
        self.votes_time_ago = np.zeros((self.num_agents, self.num_agents), dtype=int) if self.agent_removal == 'voting' else None

        self._votes_pre_reset = np.copy(self.votes)

        # Create a random permutation for the assignment of observable ids
        perm = np.random.permutation(self.num_agents)
        self.observable_id_assignment = {id: p for id, p in zip(self.agents.keys(), perm)}

        # Put coins at random positions
        coins_positions = self.map.sample_random_coordinates(self.num_coins, False)
        for p in coins_positions:
            self.coins.append(Coin(tuple(p)))

        # Put agents at random positions
        agents_positions = self.map.sample_random_coordinates(self.num_agents, False)
        for i, agent in enumerate(self.agents.values()):
            agent.pos = agents_positions[i]

        return self.get_agents_observations()

    def get_map_encoded_with_coins(self):
        map_encoded = self.map.encode(self.map.data)
        for coin in self.coins:
            vec = [0,1,0] if not coin.broken else [0,0,coin.broken_freshness]
            map_encoded[coin.pos] = vec
        return map_encoded

    def get_agents_observations(self):
        map_encoded = self.get_map_encoded_with_coins()

        views = {
            # Slice the map view for each agent
            id: {'map': self.map.get_view(map_encoded, agent.pos, agent.view_range, 0)} for id, agent in self.agents.items()
        }

        # Agents see an id for every other agent
        # The ids are 0 to self.num_agents-1. The value self.num_agents means "no agent here"
        agents_map = np.ones((self.map.shape[0], self.map.shape[1]), dtype=int) * self.num_agents
        for id, agent in self.agents.items():
            agents_map[tuple(agent.pos)] = self.observable_id_assignment[id]

        # Agents which are not removed, permutated
        playing_vec_perm = np.zeros(self.num_agents, dtype=int)
        for id in self.agents:
            playing_vec_perm[self.observable_id_assignment[id]] = 1

        # Get the permutated vector of who is a traitor and who is not
        all_agents = {**self.agents, **self.removed_agents}
        traitors_vec_perm = np.zeros(self.num_agents, dtype=int)
        for id in all_agents:
            if all_agents[id].traitor:
                traitors_vec_perm[self.observable_id_assignment[id]] = 1

        # Assign observations to each agent
        for id, agent in self.agents.items():
            # Slice the agent view for each agent
            views[id]['map_agents'] = self.map.get_view(agents_map, agent.pos, agent.view_range, self.num_agents).tolist()
            views[id]['playing'] = playing_vec_perm.tolist()
            views[id]['own_id'] = self.observable_id_assignment[id]
            # For traitors, add the vector encoding who a traitor is
            if agent.traitor:
                views[id]['traitors'] = traitors_vec_perm.tolist()
        
        if self.observable_votes:
            # If votes are observable, we add them to the observations
            # We first have to permute the votes vector again
            votes_perm = [np.zeros(self.num_agents, dtype=int).tolist() for _ in range(self.num_agents)]
            for id_from, votes in enumerate(self.votes):
                for id_for, v in enumerate(votes):
                    if v:
                        votes_perm[self.observable_id_assignment[id_from]][self.observable_id_assignment[id_for]] = 1
            
            for id in self.agents:
                views[id]['votes'] = votes_perm
        return views

    def init_agents(self):
        """
        Create the agent objects
        """
        for i in range(self.num_agents):
            is_t = i < self.num_traitors
            self.agents[i] = GridworldAgent(
                i, (0, i), self.traitors_view_range if is_t else self.innocents_view_range, self.map, is_t
            )
    
    def render(self, mode='print'):
        if mode == 'print':
            map = np.copy(self.map.data)
            print()
            print('Votes:')
            print(self._votes_pre_reset)
            for coin in self.coins:
                map[tuple(coin.pos)] = RENDERING_COIN if not coin.broken else RENDERING_BROKEN_COIN

            for id, agent in self.agents.items():
                map[tuple(agent.pos)] = id
            print(pd.DataFrame(map))

        if mode == 'image':
            GRID_SCALE = 50 # How many pixels per grid cell
            img = (self.get_map_encoded_with_coins() * 255).astype("uint8")
            vote_img = np.zeros_like(img)

            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if self._votes_pre_reset[i,j]:
                        vote_img[i,j,1] = 255
                    else:
                        vote_img[i,j,2] = 255
            
            img = cv2.resize(img, (self.map.width * GRID_SCALE, self.map.height * GRID_SCALE), interpolation=cv2.INTER_NEAREST)
            for id, agent in self.agents.items():
                cv2.putText(img, str(id), ((agent.pos[1]) * GRID_SCALE, (agent.pos[0] + 1) * GRID_SCALE), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            vote_img = cv2.resize(vote_img, (self.map.width * GRID_SCALE, self.map.height * GRID_SCALE), interpolation=cv2.INTER_NEAREST)
            return cv2.hconcat([vote_img, img])


    def get_observation_spaces(self):
        """
        Given a configuration, this method returns the correct observations spaces for the agents
        """
        base_dict = {
            'own_id': gym.spaces.Discrete(self.num_agents),
            'playing': gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(self.num_agents)])
        }

        if self.observable_votes:
            base_dict['votes'] = gym.spaces.Tuple([gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(self.num_agents)]) for _ in range(self.num_agents)])

        innocent_map_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2 * self.innocents_view_range + 1,
                                                2 * self.innocents_view_range + 1, 3), dtype=np.float32)

        traitor_map_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2 * self.traitors_view_range + 1,
                                                2 * self.traitors_view_range + 1, 3), dtype=np.float32)

        # We have a 2D discrete view for the agents. The number of discretes is num_agents+1 
        # so we either have and agent with the id from 0 to num_agents-1 or the value num_agents if no agent is on that spot
        map_agents = [gym.spaces.Tuple([gym.spaces.Tuple([gym.spaces.Discrete(self.num_agents+1) for _ in range(2 * vr + 1)]) for _ in range(2 * vr + 1)]) 
            for vr in [self.innocents_view_range, self.traitors_view_range]]

        innocents = gym.spaces.Dict({
            'map': innocent_map_observation_space,
            'map_agents': map_agents[0],
            **base_dict
        })

        traitors = gym.spaces.Dict({
            'map': traitor_map_observation_space,
            'map_agents': map_agents[1],
            'traitors': gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(self.num_agents)]),
            **base_dict
        })
        return {'innocents': innocents, 'traitors': traitors}

    def get_action_spaces(self):
        """
        Given a configuration, this method returns the correct action spaces for the agents
        """
        if self.agent_removal == 'voting':
            actions = gym.spaces.Dict({
                'move': gym.spaces.Discrete(5), # 4 directions + no-op
                'vote': gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(self.num_agents)]) # RLlib doesn't support multibinary
            })
            return {'innocents': actions, 'traitors': actions}

        elif self.agent_removal == 'attacking':
            innocents = gym.spaces.Dict({
                'move': gym.spaces.Discrete(6), # 4 directions + no-op + attack
            })
            # Traitors cannot attack. This is to balance the game since otherwise traitors could easily remove all innocents without repercussions
            # Might be interesting to experiment with this. See future work
            traitors = gym.spaces.Dict({
                'move': gym.spaces.Discrete(5), # 4 directions + no-op
            })
            return {'innocents': innocents, 'traitors': traitors}

