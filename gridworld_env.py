from typing import Dict

import numpy as np
from ray.rllib.env import MultiAgentEnv

from gridworld_agent import GridworldAgent, MOVEMENT_ACTIONS_VECTORS

CHAR_TO_VEC = {
    ' ': [0, 0, 0],  # Background
    '0': [1, 0, 0],  # Wall
}

class Map:

    """
    Map. (0,0) is top left.
    """
    def __init__(self, initial_data: np.ndarray, layer_map=None):
        """
        :param initial_data: Optional initialisation of map
        :param layer_map: Maps char to according layer, e.g. {'A': 0, 'C': 1}
        """

        self.data = np.copy(initial_data)
        self.initial_data = initial_data

        self.height, self.width = initial_data.shape
        self.layer_map = layer_map
    
    @property
    def shape(self):
        return self.data.shape

    def reset(self):
        """
        Reset the world
        """
        self.data = np.copy(self.initial_data)

    def encode(self, map):
        # Transform the character map to an rgb map so it is in line with the observation space of agents
        ret = np.zeros((len(map[0]), len(map), 3), dtype=np.float32)
        for row_i in range(ret.shape[0]):
            for col_i in range(ret.shape[1]):
                ret[row_i, col_i] = CHAR_TO_VEC[map[row_i, col_i]]
        return ret

    def to_one_hot(self, data):
        """
        :return: Returns a 3D one-hot encoding of the 2D map.
        """
        print(data)
        encoding = np.zeros(shape=(self.height, self.width, len(self.layer_map)), dtype=np.uint8)
        for i, row in enumerate(data):
            for j, column in enumerate(row):
                # TODO: Generalise and  make more efficient if possible
                if self.data[i][j] != ' ':
                    encoding[i, j, self.layer_map[self.data[i][j]]] = 1
        return encoding

    def get_view(self, map, pos, range, pad=0):
        vr = range
        pad_amts = [(vr, vr), (vr, vr)]
        if len(map.shape) > 2:
            pad_amts.append((0,0))
        # We add padding to the map, so the agent sees something when at the edge
        map = np.pad(map, pad_amts, mode='constant', constant_values=pad)
        # Adjust the position based on padding
        pos_adj = (pos[0]+vr, pos[1]+vr)
        # Slice the corresponding part from the map
        return map[pos_adj[0]-vr:pos_adj[0]+vr+1,:][:,pos_adj[1]-vr:pos_adj[1]+vr+1]

    def __getitem__(self, arg):
        return self.data[arg]

    def __setitem__(self, arg, item):
        self.data[arg] = item

    def sample_random_coordinates(self, n, replace=True):
        """
        Samples n random coordinates from the map and returns them as numpy array of tuples.
        If replace is False, all coordinates will be unique
        """
        coords = np.array(list(np.ndindex(self.shape)))
        return coords[np.random.choice(coords.shape[0], n, replace=replace)]


INITIAL_MAP = np.array(
                [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']])

class GridworldEnv(MultiAgentEnv):
    """
    Base class for multi agent gridworlds
    """

    def __init__(self, env_config):
        # Grab config parameters
        self.map = Map(INITIAL_MAP, env_config.get('layer_map'))
        self.view_ranges = env_config['view_ranges']
        self.num_agents = env_config['num_agents']
        self.rewards = env_config['rewards']

        self.dones = set()
        self.agents: Dict[GridworldAgent] = {}
        self.init_agents()

    def reset(self):
        self.dones = set()
        self.reset_agents()
        self.map.reset()


    def get_agents_observations(self):
        raise NotImplementedError()

    def step(self, action_dict):
        raise NotImplementedError()

    def move_agents(self, actions):
        """
        Move agents depending on their actions from timestep t to t+1
        An agent can only move to a new cell if:
        1. No other agent occupied this cell in timestep t
        2. No other agent occupies this cell in timestep t+1, there is at most one agent moving into this cell
        3. The cell is in map bounds
        """
        current_positions = {id: agent.pos for id, agent in self.agents.items()}
        next_raw_positions = {
            id: (self.agents[id].pos + MOVEMENT_ACTIONS_VECTORS[action]) if action in MOVEMENT_ACTIONS_VECTORS else self.agents[id].pos
            for id, action in actions.items()
        }

        next_positions = {}
        for id, agent in self.agents.items():
            next_position = next_raw_positions[id]
            if (
                (np.array(list(current_positions.values())) == next_position).all(axis=1).any()
                or (np.array(list(next_raw_positions.values())) == next_position).all(axis=1).sum() > 1
                or not self.position_valid(next_position)
            ):
                next_position = current_positions[id]

            next_positions[id] = next_position

        for id, agent in self.agents.items():
            agent.pos = next_positions[id]

    def position_valid(self, position):
        """
        Check if a position is in map bounds
        """
        return 0 <= position[0] < self.map.shape[0] and 0 <= position[1] < self.map.shape[1]

    def init_agents(self):
        """
        Create the agent objects
        """
        for i in range(self.num_agents):
            self.agents[i] = GridworldAgent(
                i, (0, i), self.view_ranges[i], self.map
            )

    def reset_agents(self):
        for i, agent in enumerate(self.agents.values()):
            agent.pos = (0, i)

