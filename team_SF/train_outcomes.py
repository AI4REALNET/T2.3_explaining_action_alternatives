import numpy as np

import gymnasium as gym

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from team_SF.trajectory_buffer import MCReplayBuffer
from team_SF.explanation_task import get_multi_user_menu_choice_likelihood

from tqdm import trange

import wandb
import time

def normalize_log_belief(x):
    return x - x.max() - np.log(np.exp(x - x.max()).sum())


def collect_trajectories(env, DM_types, n_steps, buffer, policy, explanation):
    N = 0

    while N < n_steps:
        s_vec, b_vec, a_vec, e_vec, ua_vec, r_vec, fsa_vec = [], [], [], [], [], [], []

        s, _ = env.reset()
        s_sub = s["sub state"]
        done = False
        log_belief = normalize_log_belief(np.zeros(DM_types.shape[0], np.float32))

        while not done:
            a = policy(s_sub, log_belief)
            expl = explanation(np.array(gym.spaces.utils.flatten(env.sub_env.observation_space, s_sub), dtype=np.float32), log_belief, a)

            s_p, r, terminated, truncated, _ = env.step(a, expl)

            user_choice = np.unravel_index(s_p["prev choice"].argmax(), s_p["prev choice"].shape)

            s_vec.append(gym.spaces.utils.flatten(env.sub_env.observation_space, s_sub))
            b_vec.append(log_belief)
            a_vec.append(a)
            e_vec.append(expl)
            ua_vec.append(s_p["prev choice"])
            r_vec.append(np.dot(r, env.theta))
            fsa_vec.append(r)

            log_belief += get_multi_user_menu_choice_likelihood(a, expl, DM_types, env.beta, user_choice)
            log_belief = normalize_log_belief(log_belief)

            s = s_p
            s_sub = s["sub state"]
            done = terminated or truncated
            N += 1

        buffer.append(np.array(s_vec), np.array(b_vec), np.array(a_vec), np.array(e_vec), np.array(ua_vec), np.array(r_vec), np.array(fsa_vec))


def eval_return_ep_length(env, DM_types, n_eps, policy, explanation):
    ep_returns = []
    N = 0

    for _ in range(n_eps):
        s, _ = env.reset()
        s_sub = s["sub state"]
        done = False
        log_belief = normalize_log_belief(np.zeros(DM_types.shape[0], np.float32))

        ep_return = 0
        n_a = 0

        while not done:
            a = policy(s_sub, log_belief)
            expl = explanation(np.array(gym.spaces.utils.flatten(env.sub_env.observation_space, s_sub), dtype=np.float32), log_belief, a)

            s_p, r, terminated, truncated, _ = env.step(a, expl)

            user_choice = np.unravel_index(s_p["prev choice"].argmax(), s_p["prev choice"].shape)

            ep_return += np.dot(r, env.theta) * env.discount_rate ** n_a
            n_a += (np.sum(a[:,0]) + np.sum(np.logical_and(a[:,0], a[:,1])))

            log_belief += get_multi_user_menu_choice_likelihood(a, expl, DM_types, env.beta, user_choice)
            log_belief = normalize_log_belief(log_belief)

            s = s_p
            s_sub = s["sub state"]
            done = terminated or truncated
            N += 1

        ep_returns.append(ep_return)
    
    return {"DM_return_val": np.mean(ep_returns), "ep_length_val": N / n_eps}


def train_outcome_predictor(Eenv, DM_types, opred, action_selection_policy, total_n_steps = 1_000_000, n_epochs = 10, n_steps = 2048, batch_size = 64, lr_start = 1e-4, lr_end_factor = 1.0, eval_callback = None, callback_freq = 1, log_to_wandb = False):
    """
    Trains an outcome predictor.

    Arguments:
    Eenv -- an explanation environment of the type team_SF.explanatino_task.ExplanationEnv
    DM_types -- an (N_user_types X N_outcomes) vector of weights for each user type considered
    opred -- an outcome prediction model of the type team_SF.models.OutcomePredictor
    action_selection_policy -- a given policy which selects which actions to explain in each state
    total_n_steps -- over how mamy total interactions with the environment to train
    n_epochs -- number of epochs to train after each new collection of interactions with the environment
    n_steps -- number of environment interactions to collect each time collection is run
    batch_size -- size of mini-batch in each epoch (default: 64)
    lr_start -- learning rate at the start of training (default: 1e-4)
    lr_end_factor -- by how much to decrease the learning rate as training progresses; the final learning rate will be lr_start * lr_end_factor (default: 1.0)
    eval_callback -- function to call periodically to evaluate training progress
    callback_freq -- how often to run the eval_callback. The callback will be run every n_epochs * callback_freq minibatch updates.
    """
    collection_time = 0
    training_time = 0

    n_collections = total_n_steps // n_steps

    def pred_explanation(state, log_belief, action):
        # generate explanations using model
        s_batch = torch.from_numpy(state)[None,...]
        b_batch = torch.from_numpy(log_belief)[None,...]
        a_batch = torch.from_numpy(action)[None,...]
        return opred(s_batch, b_batch, a_batch)[0,...].detach().numpy()

    buffer = MCReplayBuffer(discounting=Eenv.discount_rate)
    optimizer = AdamW(opred.parameters(), lr=lr_start)
    scheduler = ExponentialLR(optimizer, np.exp(np.log(lr_end_factor) / n_collections))

    for collection_step in trange(n_collections):
        with torch.no_grad():
            buffer.reset()
            start_collection_time = time.time()
            collect_trajectories(Eenv, DM_types, n_steps, buffer, action_selection_policy, pred_explanation)
            collection_time += (time.time() - start_collection_time)

        for epoch in range(n_epochs):
            start_epoch = time.time()
            s_batch, b_batch, a_batch, _, ua_batch, _, fsa_batch = buffer.sample_batch_MC(batch_size)

            pred = opred(torch.from_numpy(s_batch), torch.from_numpy(b_batch), torch.from_numpy(a_batch))
            mask = torch.from_numpy(ua_batch)[...,None].repeat_interleave(pred.shape[-1], -1)
            TDerr = torch.nn.functional.smooth_l1_loss((pred * mask).sum(dim=[1,2]), torch.from_numpy(fsa_batch))

            # backward pass
            optimizer.zero_grad()
            TDerr.backward()

            # weight update
            optimizer.step()

            training_time += (time.time() - start_epoch)

            train_metrics = {"SF_err": TDerr.detach().item()}

            if (collection_step % callback_freq == 0) and (epoch == n_epochs-1) and (eval_callback is not None):
                with torch.no_grad():
                    val_dict = eval_callback(Eenv, DM_types, action_selection_policy, pred_explanation)
                    train_metrics.update(val_dict)
                if not log_to_wandb:
                    print(val_dict)
            if log_to_wandb:
                wandb.log(train_metrics)
        scheduler.step()

    final_metrics = {"train_time": training_time, "rollout_time": collection_time}
    if eval_callback is not None:
        final_metrics.update(eval_callback(Eenv, DM_types, action_selection_policy, pred_explanation))
    if log_to_wandb:
        wandb.log(final_metrics)
    else:
        print(final_metrics)