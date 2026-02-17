from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete


class DamEnv(gym.Env):
    """
    ## Description
    A Water reservoir environment.
    The agent executes a continuous action, corresponding to the amount of water released by the dam.

    A. Castelletti, F. Pianosi and M. Restelli, "Tree-based Fitted Q-iteration for Multi-Objective Markov Decision problems,"
    The 2012 International Joint Conference on Neural Networks (IJCNN),
    Brisbane, QLD, Australia, 2012, pp. 1-8, doi: 10.1109/IJCNN.2012.6252759.

    ## Observation Space
    The observation is a float corresponding to the current level of the reservoir.

    ## Action Space
    The action is a float corresponding to the amount of water released by the dam.
    If normalized_action is True, the action is a float between 0 and 1 corresponding to the percentage of water released by the dam.

    ## Reward Space
    There are up to 4 rewards:
     - cost due to excess level wrt a flooding threshold (upstream)
     - deficit in the water supply wrt the water demand
     - deficit in hydroelectric supply wrt hydroelectric demand
     - cost due to excess level wrt a flooding threshold (downstream)
     By default, only the first two are used.

     ## Starting State
     The reservoir is initialized with a random level between 0 and 160.

     ## Arguments
        - render_mode: The render mode to use. Can be 'human', 'rgb_array' or 'ansi'.
        - time_limit: The maximum number of steps until the episode is truncated.
        - rain_inflow: If true, inflow is modeled as rain using heavy tails, if False inflow is modeled using the original Normal Distribution from Mathieu Reymond.
        - nO: The number of objectives to use. Can be 2, 3 or 4.
        - penalize: Whether to penalize the agent for selecting an action out of bounds.
        - normalized_action: Whether to normalize the action space as a percentage [0, 1].
        - initial_state: The initial state of the reservoir. If None, a random state is used.
        - scale_rewards: If True, rescale rewards to [0,1]
     ## Credits
     Code from:
     [Mathieu Reymond](https://gitlab.ai.vub.ac.be/mreymond/dam).
     Ported from:
     [Simone Parisi](https://github.com/sparisi/mips).
    """

    W_IRR = 50.0  # Water demand
    H_FLO_U = 300.0  # Flooding threshold (upstream, i.e. height of dam)
    DAM_INFLOW_MEAN = 40.0  # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 10.0
    Q_MEF = 0.0
    GAMMA_H2O = 1000.0  # water density
    W_HYD = 4.36  # Hydroelectric demand
    Q_FLO_D = 100.0  # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1.0  # Turbine efficiency
    G = 9.81  # Gravity

    utopia = {2: [-0.5, -9], 3: [-0.5, -9, -0.0001], 4: [-0.5, -9, -0.001, -9]}
    antiutopia = {2: [-2.5, -11], 3: [-65, -12, -0.7], 4: [-65, -12, -0.7, -12]}

    # Create colors.
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    s_init = np.array(
        [
            9.6855361e01,
            5.8046026e01,
            1.1615767e02,
            2.0164311e01,
            7.9191000e01,
            1.4013098e02,
            1.3101816e02,
            4.4351321e01,
            1.3185943e01,
            7.3508622e01,
        ],
        dtype=np.float32,
    )

    metadata = {"render_modes": ["ansi"], "render_fps": 2}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_limit: int = 100,
        rain_inflow: bool = False,
        nO=2,
        penalize: bool = False,
        n_discr_actions: int = 11,
        initial_state: Optional[np.ndarray] = None,
        vectorize_reward = None,
        scale_rewards = None,
        include_timestep = False,
    ):
        self.render_mode = render_mode
        self.scale_rewards = scale_rewards
        self.vectorized_reward = vectorize_reward
        self.rain_inflow = rain_inflow

        self.include_timestep = include_timestep
        if include_timestep:
            self.observation_space = Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32)
        else:
            self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = Discrete(n_discr_actions)
        self.actions = np.linspace(0.0, 1.0, n_discr_actions)

        self.nO = nO
        self.penalize = penalize
        self.time_limit = time_limit
        self.initial_state = initial_state
        self.time_step = 0
        self.last_action = None
        self.dam_inflow = None
        self.excess = None
        self.defict = None

        low = -np.ones(nO, dtype=np.float32) * np.inf
        high = np.zeros(nO, dtype=np.float32)
        self.reward_space = Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.reward_dim = nO

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time_step = 0
        if self.initial_state is not None:
            self.state = self.initial_state
        else:
            self.state = np.array(self.np_random.integers(20, 140, size=1), dtype=np.float32)

        if self.include_timestep:
            obs = np.concatenate([self.state, np.array([self.time_step])])
        else:
            obs = self.state

        return obs, {}

    def render(self):
        assert self.render_mode == "ansi"

        outfile = StringIO()
        outfile.write(f"Water level: {self.state[0]:.2f}\n")
        if self.last_action is not None:
            outfile.write(f"Water released: {self.last_action:.2f}\n")
            outfile.write(f"Dam inflow: {self.dam_inflow:.2f}\n")
            outfile.write(f"Demand deficit: {self.defict:.2f}\n")
            outfile.write(f"Flooding excess: {self.excess:.2f}\n")

        with closing(outfile):
            return outfile.getvalue()

    def step(self, action_i):
        # lookup normalized action in self.actions and calculate water release
        action = self.actions[action_i] * DamEnv.Q_FLO_D # ACTION == OUTFLOW!!!

        # transition dynamic
        self.last_action = action
        if self.rain_inflow:
            # Parameters follow "Geng, S., Penning de Vries, F.W.T. and Supit, I., 1985., 1985. A simple method for generating rainfall data. Agric. For. Meteorol., 36 : 363--376." and assume 75% wet days.
            self.dam_inflow = self.np_random.binomial(1, 0.75) * self.np_random.gamma(0.6, 20*4)
        else:
            self.dam_inflow = self.np_random.normal(DamEnv.DAM_INFLOW_MEAN, DamEnv.DAM_INFLOW_STD)
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + self.dam_inflow - action, 0, None).astype(np.float32)
        outflow = np.array(min(self.state[0] + self.dam_inflow, action))

        # cost due to excess level wrt a flooding threshold (upstream)
        self.excess = np.clip(n_state[0] - DamEnv.H_FLO_U, 0, None)
        r0 = -self.excess
        # deficit in the water supply wrt the water demand
        self.defict = np.clip(DamEnv.W_IRR - outflow, 0, None)
        r1 = -self.defict

        q = np.clip(outflow - DamEnv.Q_MEF, 0, None)
        p_hyd = DamEnv.ETA * DamEnv.G * DamEnv.GAMMA_H2O * n_state[0] * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(DamEnv.W_HYD - p_hyd, 0, None)
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(outflow - DamEnv.Q_FLO_D, 0, None)

        reward = (np.array([r0, r1, r2, r3], dtype=np.float32))[: self.nO].flatten()
        if self.scale_rewards is not None:
            reward *= self.scale_rewards[: self.nO]

        self.state = n_state

        self.time_step += 1
        truncated = self.time_step >= self.time_limit
        terminated = False

        if self.include_timestep:
            obs = np.concatenate([n_state, np.array([self.time_step])])
        else:
            obs = n_state
        
        if self.vectorized_reward is not None:
            return obs, np.sum(reward * self.vectorized_reward), terminated, truncated, {}

        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = DamEnv(nO = 3, vectorize_reward=None, rain_inflow=True)
    obs, info = env.reset()
    N = 0
    R = np.zeros(3)
    returns = []
    while N < 1000:
        action = env.state
        obs, reward, terminated, truncated, info = env.step(np.random.randint(0,11))
        R += reward
        if terminated or truncated:
            obs, info = env.reset()
            N += 1
            returns.append(R)
            R = 0
    print(np.mean(returns, axis=0), np.std(returns, axis=0), np.max(returns, axis=0), np.min(returns, axis=0))
