import argparse

import numpy as np
import torch

import matplotlib.pyplot as plt

from envs.mo_gridworld import MOGridWorld
from envs.dam import DamEnv
from team_SF.explanation_task import ExplanationEnv, get_multi_user_menu_choice_likelihood
from team_SF.train_outcomes import train_outcome_predictor, eval_return_ep_length, normalize_log_belief
from team_SF.models import OutcomePredictor, StateEmbedder, ActionEmbedder, ZeroEmbedder

import gymnasium as gym

import wandb

def eval_outcome_predictor(Eenv, DM_types, n_eps, policy, explanation):
    t_expl_error = []
    t_posterior_entropy = []

    for _ in range(n_eps):
        s, _ = Eenv.reset()
        s_sub = s["sub state"]
        done = False
        log_belief = normalize_log_belief(np.zeros(DM_types.shape[0], np.float32))

        ep_return = 0
        n_a = 0

        selected_explanations = []
        rs = []
        log_beliefs = [log_belief.copy()]

        while not done:
            a = policy(s_sub, log_belief)
            expl = explanation(np.array(gym.spaces.utils.flatten(Eenv.sub_env.observation_space, s_sub), dtype=np.float32), log_belief, a)

            s_p, r, terminated, truncated, _ = Eenv.step(a, expl)

            user_choice = np.unravel_index(s_p["prev choice"].argmax(), s_p["prev choice"].shape)

            ep_return += np.dot(r, Eenv.theta) * Eenv.discount_rate ** n_a
            n_a += (np.sum(a[:,0]) + np.sum(np.logical_and(a[:,0], a[:,1])))

            log_belief += get_multi_user_menu_choice_likelihood(a, expl, DM_types, Eenv.beta, user_choice)
            log_belief = normalize_log_belief(log_belief)

            selected_explanations.append(expl[*user_choice])
            rs.append(r)
            log_beliefs.append(log_belief.copy())

            s = s_p
            s_sub = s["sub state"]
            
            done = terminated or truncated

        # calculate explanation errors
        rs = np.array(rs)
        true_outcomes = [np.sum(np.multiply(rs[t:,...], np.power(Eenv.discount_rate, np.arange(0, rs.shape[0]-t, 1))[:,None].repeat(rs.shape[1],-1)), axis=0) for t in range(rs.shape[0])]
        t_expl_error.append(np.sqrt(np.power(np.array(true_outcomes) - np.array(selected_explanations), 2).sum(axis=1)))

        # calculate posterior entropy
        log_beliefs = np.array(log_beliefs)
        t_posterior_entropy.append(np.sum(- np.exp(log_beliefs) * log_beliefs, axis=1))

    def transform_to_per_t(ep_list):
        """
        Transform an ep_list from a list of episodes (consisting of vectors of timesteps) to a list of timesteps (consisting of a list of episodes)
        """
        max_ep_length = np.max([ep.shape[0] for ep in ep_list])
        data_per_t = [[] for _ in range(max_ep_length)]
        for ep in ep_list:
            for i,x in enumerate(ep):
                data_per_t[i].append(x)
        return data_per_t
    
    return transform_to_per_t(t_posterior_entropy), transform_to_per_t(t_expl_error)

def train_test_gridworld():
    run_config = dict(
        n_actions = 4,
        n_belief = 4,
        s_emb_size = 16,
        a_emb_size = 8,
        env_discount_rate = 0.95,
        DM_beta = 20.0,
        state_embedder_layers = [64],
        action_embedder_layers = [32],
        outcome_predictor_layers = [64, 64, 64],
        total_n_steps = 1_000_000,
        n_epochs = 4,
        n_steps = 512,
        lr_start = 1e-5,
        lr_end_factor = 1.0,
        n_eval_eps = 200
    )

    user_space = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def sample_user_weights():
        return user_space[np.random.randint(0,4),:]
    
    get_env = lambda: MOGridWorld(backup=True, backup_penalty = 0.5, horizon = 50)
    Eenv = ExplanationEnv(get_env(), sample_user_weights, run_config["DM_beta"], discount_rate=run_config["env_discount_rate"])

    def action_selection_policy(state, log_belief):
        if state == 0:
            return np.array([[0,0],[1,0],[1,0],[0,0]], dtype=np.float32)
        elif state in [1,3]:
            return np.array([[1,0],[1,0],[1,0],[1,0]], dtype=np.float32)
        elif state == 2:
            return np.array([[0,0],[1,0],[1,0],[1,0]], dtype=np.float32)
        elif state == 4:
            return np.array([[0,0],[0,0],[1,0],[1,0]], dtype=np.float32)
        elif state in [5,6,7,8,12]:
            return np.array([[1,0],[0,0],[0,0],[0,0]], dtype=np.float32)
        elif state == 9:
            return np.array([[0,0],[0,0],[0,0],[1,0]], dtype=np.float32)
        elif state == 10:
            return np.array([[1,1],[1,0],[0,0],[1,0]], dtype=np.float32)
        elif state == 11:
            return np.array([[0,0],[1,0],[0,0],[0,0]], dtype=np.float32)
        else:
            raise NotImplementedError
        
    opred = OutcomePredictor(StateEmbedder(13, run_config["n_belief"], run_config["s_emb_size"], run_config["state_embedder_layers"]), ActionEmbedder(run_config["n_actions"], run_config["a_emb_size"], run_config["action_embedder_layers"]), run_config["s_emb_size"], run_config["a_emb_size"], run_config["n_actions"], 3, run_config["outcome_predictor_layers"])

    wandb.init(project="TeamSF", tags = [get_env().__class__.__qualname__], config = run_config)

    train_outcome_predictor(Eenv, user_space, opred, action_selection_policy,
                            total_n_steps = run_config["total_n_steps"],
                            n_epochs = run_config["n_epochs"],
                            n_steps = run_config["n_steps"],
                            lr_start = run_config["lr_start"],
                            lr_end_factor = run_config["lr_end_factor"],
                            eval_callback = lambda e, t, p, x: eval_return_ep_length(e, t, run_config["n_eval_eps"], p, x),
                            callback_freq=100, log_to_wandb=True)
    
    def pred_explanation(state, log_belief, action):
        # generate explanations using model
        s_batch = torch.from_numpy(state)[None,...]
        b_batch = torch.from_numpy(log_belief)[None,...]
        a_batch = torch.from_numpy(action)[None,...]
        return opred(s_batch, b_batch, a_batch)[0,...].detach().numpy()
    
    posterior_entropy_per_t, expl_error_per_t = eval_outcome_predictor(Eenv, user_space, 1000, action_selection_policy, pred_explanation)

    # do not plot timesteps reached by fewer than 10% of episodes.
    for t_cutoff, n_eps in enumerate([len(t) for t in expl_error_per_t]):
        if n_eps < 1000/10:
            break

    post_per_t_mean = np.array([np.mean(t) for t in posterior_entropy_per_t[:t_cutoff]])
    post_per_t_std = np.array([np.std(t) for t in posterior_entropy_per_t[:t_cutoff]])
    post_per_t_stderr = np.array([np.std(t) / np.sqrt(len(t)) for t in posterior_entropy_per_t[:t_cutoff]])
    plt.errorbar(np.arange(0,post_per_t_mean.shape[0]), post_per_t_mean, yerr=2 * post_per_t_stderr, label = "mean ± 2 * std. err.")
    plt.fill_between(np.arange(0,post_per_t_mean.shape[0]), post_per_t_mean - 2 * post_per_t_std, post_per_t_mean + 2 * post_per_t_std, alpha = 0.3, label = "mean ± 2 *  std. dev.")
    plt.title("Uncertainty about decision maker type over time")
    plt.legend()
    plt.show()

    expl_error_per_t_mean = np.array([np.mean(t) for t in expl_error_per_t[:t_cutoff]])
    expl_error_per_t_std = np.array([np.std(t) for t in expl_error_per_t[:t_cutoff]])
    expl_error_per_t_stderr = np.array([np.std(t) / np.sqrt(len(t)) for t in expl_error_per_t[:t_cutoff]])
    plt.errorbar(np.arange(0,expl_error_per_t_mean.shape[0]), expl_error_per_t_mean, yerr=2 * expl_error_per_t_stderr, label = "mean ± 2 * std. err.")
    plt.fill_between(np.arange(0,expl_error_per_t_mean.shape[0]), expl_error_per_t_mean - 2 * expl_error_per_t_std, expl_error_per_t_mean + 2 * expl_error_per_t_std, alpha = 0.3, label = "mean ± 2 *  std. dev.")
    plt.title("Empirical error in predicted outcome over time")
    plt.legend()
    plt.show()

def train_test_dam(log_to_wandb = False, include_t = True):
    run_config = dict(
        n_actions = 7,
        n_belief = 128,
        s_emb_size = 16,
        a_emb_size = 32,
        env_discount_rate = 0.99,
        DM_beta = 20.0,
        state_embedder_layers = [64,64],
        action_embedder_layers = None,
        outcome_predictor_layers = [64, 64, 64],
        total_n_steps = 25_000_000,
        n_epochs = 1,
        n_steps = 512,
        batch_size = 128,
        lr_start = 1e-4,
        lr_end_factor = 0.1,
        n_eval_eps = 200
    )

    def sample_user_weights():
        w = np.power(10, - np.random.rand(3) * 4)
        return w / w.sum()

    def action_selection_policy(state, log_belief):
        return np.array([[1,0]] * run_config["n_actions"], dtype=np.float32)
    
    get_env = lambda : DamEnv(nO = 3, n_discr_actions = run_config["n_actions"], scale_rewards = np.array([1/400, 1/50, 1/30, 0.0]), include_timestep = include_t)
    Eenv = ExplanationEnv(get_env(), sample_user_weights, run_config["DM_beta"], discount_rate=run_config["env_discount_rate"])

    user_space = np.array([sample_user_weights() for _ in range(run_config["n_belief"])])

    opred = OutcomePredictor(StateEmbedder(2 if include_t else 1, run_config["n_belief"], run_config["s_emb_size"], run_config["state_embedder_layers"]), ZeroEmbedder(run_config["a_emb_size"]), run_config["s_emb_size"], run_config["a_emb_size"], run_config["n_actions"], 3, run_config["outcome_predictor_layers"])

    if log_to_wandb:
        wandb.init(project="TeamSF", tags = [get_env().__class__.__qualname__ + f"_{run_config["n_actions"]}" + "_T" if include_t else ""], config = run_config)

    train_outcome_predictor(Eenv, user_space, opred, action_selection_policy,
                            total_n_steps = run_config["total_n_steps"],
                            n_epochs = run_config["n_epochs"],
                            n_steps = run_config["n_steps"],
                            lr_start = run_config["lr_start"],
                            lr_end_factor = run_config["lr_end_factor"],
                            batch_size = run_config["batch_size"],
                            eval_callback = lambda e, t, p, x: eval_return_ep_length(e, t, run_config["n_eval_eps"], p, x),
                            callback_freq=5000,
                            log_to_wandb = log_to_wandb)
    
    if log_to_wandb:
        id = wandb.util.generate_id()
        torch.save(opred.state_dict(), f"checkpoints/Dam/{id}.pth")
        np.save(f"checkpoints/Dam/{id}_DM_types.npy", user_space)
        wandb.config.checkpoint = f"checkpoints/Dam/{id}.pth"

    def pred_explanation(state, log_belief, action):
        # generate explanations using model
        s_batch = torch.from_numpy(state)[None,...]
        b_batch = torch.from_numpy(log_belief)[None,...]
        a_batch = torch.from_numpy(action)[None,...]
        return opred(s_batch, b_batch, a_batch)[0,...].detach().numpy()

    posterior_entropy_per_t, expl_error_per_t = eval_outcome_predictor(Eenv, user_space, 1000, action_selection_policy, pred_explanation)

    post_per_t_mean = np.array([np.mean(t) for t in posterior_entropy_per_t])
    post_per_t_std = np.array([np.std(t) for t in posterior_entropy_per_t])
    post_per_t_stderr = np.array([np.std(t) / np.sqrt(len(t)) for t in posterior_entropy_per_t])
    plt.errorbar(np.arange(0,post_per_t_mean.shape[0]), post_per_t_mean, yerr=2 * post_per_t_stderr, label = "mean ± 2 * std. err.")
    plt.fill_between(np.arange(0,post_per_t_mean.shape[0]), post_per_t_mean - 2 * post_per_t_std, post_per_t_mean + 2 * post_per_t_std, alpha = 0.3, label = "mean ± 2 *  std. dev.")
    plt.title("Uncertainty about decision maker type over time")
    plt.legend()
    plt.show()

    expl_error_per_t_mean = np.array([np.mean(t) for t in expl_error_per_t])
    expl_error_per_t_std = np.array([np.std(t) for t in expl_error_per_t])
    expl_error_per_t_stderr = np.array([np.std(t) / np.sqrt(len(t)) for t in expl_error_per_t])
    plt.errorbar(np.arange(0,expl_error_per_t_mean.shape[0]), expl_error_per_t_mean, yerr=2 * expl_error_per_t_stderr, label = "mean ± 2 * std. err.")
    plt.fill_between(np.arange(0,expl_error_per_t_mean.shape[0]), expl_error_per_t_mean - 2 * expl_error_per_t_std, expl_error_per_t_mean + 2 * expl_error_per_t_std, alpha = 0.3, label = "mean ± 2 *  std. dev.")
    plt.title("Empirical error in predicted outcome over time")
    plt.legend()
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["MOGridWorld", "Dam"])
    args = parser.parse_args()

    if args.env == "MOGridWorld":
        train_test_gridworld()
    elif args.env == "Dam":
        train_test_dam()
    else:
        raise NotImplementedError()
