from enum import Enum

import numpy as np

class MOVEMENT_ACTIONS(Enum):
    MOVE_LEFT = 'MOVE_LEFT'
    MOVE_RIGHT = 'MOVE_RIGHT'
    MOVE_UP = 'MOVE_UP'
    MOVE_DOWN = 'MOVE_DOWN'
    NOP = 'NOP'
    ATTACK = 'ATTACK'
# Attack is also counted as a movement so that agents must stand still when attacking

MOVEMENT_ACTIONS_VECTORS = {
    MOVEMENT_ACTIONS.MOVE_LEFT: np.array([-1, 0]),
    MOVEMENT_ACTIONS.MOVE_RIGHT: np.array([1, 0]),
    MOVEMENT_ACTIONS.MOVE_UP: np.array([0, -1]),
    MOVEMENT_ACTIONS.MOVE_DOWN: np.array([0, 1]),
    MOVEMENT_ACTIONS.NOP: np.array([0, 0]),
}

class GridworldAgent:

    def __init__(self, agent_id, start_pos, view_range, map, traitor=False):
        self.agent_id = agent_id
        self.pos = start_pos
        self.view_range = view_range
        self.map = map
        self.traitor = traitor

        self.movement_action_map = {i: a for i, a in enumerate(MOVEMENT_ACTIONS)}

