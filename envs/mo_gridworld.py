import numpy as np

from gymnasium.core import Env
from gymnasium.spaces import Discrete, Box

class MOGridWorld(Env):
    """
    MDP formulation of network ENV.
    """
    metadata = {"render.modes": ["ansi"]}

    N_states = 13
    N_actions = 4
    N_goals = 3

    @staticmethod
    def generate_rewards(penanlty):
        rs = np.zeros((MOGridWorld.N_states, MOGridWorld.N_actions, MOGridWorld.N_goals))
        rs[0,1,0] = 1.0
        rs[2,3,0] = 1.0
        rs[2,1,1] = 1.0
        rs[4,3,1] = 1.0
        rs[9,3,2] = 0.9
        rs[11,1,2] = 0.9

        rs[6,2,:] -= penanlty
        rs[9,1,:] -= penanlty
        rs[11,3,:] -= penanlty
        
        return rs

    def __init__(self, backup = True, backup_penalty = 0.0, horizon = 50):
        self.terminal_states = [1,3]
        if backup:
            self.transitions = np.array([[0, 1, 5, 0], [1, 1, 1, 1], [2, 3, 6, 1], [3, 3, 3, 3], [4, 4, 7, 3], [0, 5, 5, 5], [2, 6, 10, 6], [4, 7, 7, 7], [5, 8, 8, 8], [9, 10, 9, 8], [6, 11, 10, 9], [11, 12, 11, 10], [7, 12, 12, 12]])
        else:
            self.transitions = np.array([[0, 1, 5, 0], [1, 1, 1, 1], [2, 3, 6, 1], [3, 3, 3, 3], [4, 4, 7, 3], [0, 5, 5, 5], [2, 6, 6, 6], [4, 7, 7, 7], [5, 8, 8, 8], [9, 9, 9, 8], [6, 11, 10, 9], [11, 12, 11, 11], [7, 12, 12, 12]])
        self.rewards = MOGridWorld.generate_rewards(backup_penalty)

        self.action_space = Discrete(MOGridWorld.N_actions)
        self.observation_space = Discrete(MOGridWorld.N_states)
        self.reward_space = Box(0.0, 1.0, (2,))
        self.reward_dim = MOGridWorld.N_goals

        self.current_state = 10

        self.T = 0
        self.horizon = horizon

    def reward(self, state, action):
        return self.rewards[state, action, :]

    def reset(self, seed = None, options = {}):
        self.state = 10
        self.T = 0
        return self.state, {}
    
    def step(self, action):
        assert self.action_space.contains(action)

        reward = self.rewards[self.state, action, :]
        self.state = self.transitions[self.state, action]
        self.T += 1

        return self.state, reward, self.state in self.terminal_states, self.T > self.horizon, {}
    
    def render(self, mode="ansi", close=False):
        return str(self.state)