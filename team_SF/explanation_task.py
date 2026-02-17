import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import Discrete, MultiBinary, Dict

# action: N_actions_sub * 2
# explanations: N_actions_sub * 2 * N_outcomes
def get_user_menu_choice_policy(action, explanations, theta, beta):
    a_utils = np.dot(explanations, theta)
    a_logprobs = a_utils * beta

    # mask out unavailable options
    for i in range(action.shape[0]):
        if not action[i,0]:
            a_logprobs[i,:] = -np.inf
        elif not action[i,1]:
            a_logprobs[i,1] = -np.inf

    # normalize logprobs
    norm = a_logprobs.max() + np.log(np.exp(a_logprobs - a_logprobs.max()).sum())
    a_logprobs -= norm

    return a_logprobs

# action: N_actions_sub * 2
# explanations: N_actions_sub * 2 * N_outcomes
# thetas: N_users * N_outcomes
# beta: rationality parameter
# choice: tuple, index within action-explanation menu choisen
def get_multi_user_menu_choice_likelihood(action, explanations, thetas, beta, choice):
    a_logprobs = np.dot(explanations, thetas.T) * beta # N_actions x 2 x N_user_types

    # mask out unavailable options
    for i in range(action.shape[0]):
        if not action[i,0]:
            a_logprobs[i,:,:] = -np.inf
        elif not action[i,1]:
            a_logprobs[i,1,:] = -np.inf

    # normalize logprobs
    a_logprobs_max = a_logprobs.max(axis=(0,1))
    norm_logprobs = a_logprobs - a_logprobs_max - np.log(np.exp(a_logprobs - a_logprobs_max).sum(axis=(0,1)))

    return norm_logprobs[*choice,:]

class ExplanationEnv(Env):
    # action space: for each action in the sub env, the first dim of the action space of ExplanationEnv determines is the explanation is included, the second determines if a second explanation is to be shown.

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, sub_env: Env, omega_prior, beta, discount_rate = 0.99):
        self.sub_env = sub_env
        self.N_actions = None
        assert type(sub_env.action_space) is Discrete, "sub environment action space must be Discrete"
        N_actions_sub = sub_env.action_space.n
        self.action_space = MultiBinary((N_actions_sub,2))
        self.observation_space = Dict({"sub state": self.sub_env.observation_space, "prev choice": MultiBinary((N_actions_sub,2))})
        self.reward_space = self.sub_env.reward_space
        self.reward_dim = self.sub_env.reward_dim

        self.beta = beta
        self.sample_user = omega_prior

        self.state = None
        self.theta = None

        self.discount_rate = discount_rate

    def reset(self, seed = None, options = {}):
        self.state = self.sub_env.reset()[0]
        self.theta = self.sample_user()

        return {"sub state": self.state, "prev choice": np.zeros(self.observation_space["prev choice"].n, dtype=self.observation_space["prev choice"].dtype)}, {}
    
    # action: N_actions_sub * 2
    # explanations: N_actions_sub * 2 * N_outcomes
    def step(self, action, explanations):
        assert self.action_space.contains(action)

        logprobs = get_user_menu_choice_policy(action, explanations, self.theta, self.beta)
        user_choice = np.random.multinomial(1, np.exp(logprobs.flatten()) / np.exp(logprobs).sum() ).reshape(-1,2)
        chosen_action = (user_choice[:,0] | user_choice[:,1]).argmax()

        self.state, reward, terminated, truncated, info = self.sub_env.step(chosen_action)

        return {"sub state": self.state, "prev choice": user_choice}, reward, terminated, truncated, info
    
    def render(self, mode="ansi", close=False):
        return str(self.theta) + " " + str(self.state)