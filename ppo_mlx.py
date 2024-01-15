import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from torch.utils.tensorboard import SummaryWriter
from mlx_utils import *


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def loss_function(model, pg_loss, ent_coef, entropy_loss, v_loss, vf_coef):
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    return loss

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = softmax(logits)
        if action is None:
            action = sample_from_categorical(probs)

        return action, log_prob(probs, action), calc_entropy(probs), self.critic(x)


if __name__ == "__main__":
    # Experiment settings
    # exp_name = os.path.basename(__file__).rstrip(".py")
    
    # agents = ["sender"]
    learning_rate = 1e-5
    seed = 1
    total_timesteps = 250000
    # total_timesteps = 100000
    torch_deterministic = True
    cuda = True
    mps = False
    track = False
    wandb_project_name = "ppo-implementation-details"
    wandb_entity = None
    capture_video = False

    # Algorithm-specific arguments
    num_envs = 2
    num_agents = 2
    num_steps = 8192
    anneal_lr = False
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 128
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.5
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    store_freq = 20
    gym_id = "CartPole-v1"

    # Calculate derived variables
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    run_name = "first_run"
    writer = SummaryWriter(f"runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, seed + i) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs)
    optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
    loss_and_grad_fn = nn.value_and_grad(agent, loss_function)

    # ALGO Logic: Storage setup
    obs = mx.zeros((num_steps, num_envs) + envs.single_observation_space.shape)
    actions = mx.zeros((num_steps, num_envs) + envs.single_action_space.shape)
    logprobs = mx.zeros((num_steps, num_envs))
    rewards = mx.zeros((num_steps, num_envs))
    dones = mx.zeros((num_steps, num_envs))
    values = mx.zeros((num_steps, num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    next_obs = envs.reset()
    next_obs = mx.array(next_obs)
    next_done = mx.zeros(num_envs)
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        start_time = time.time()

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.tolist())
            rewards[step] = mx.array(reward)
            next_obs, next_done = mx.array(next_obs), mx.array(done)

            for item in info:
                if "episode" in item.keys():
                    # print(f"global_step={global_step}, episode_reward={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        next_value = agent.get_value(next_obs).reshape(1, -1)
        if gae:
            advantages = mx.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = mx.zeros_like(rewards)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

        # print("Inference SPS:", int(num_steps * num_envs / (time.time() - start_time)))

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = mx.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            b_inds = shuffle_array(b_inds)
            for start in range(0, batch_size, minibatch_size):
                start_loss = time.time()

                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.astype(mx.int64)[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).astype(mx.float32).mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (custom_std(mb_advantages) + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * mx.clip(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = mx.maximum(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.reshape(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + mx.clip(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = mx.maximum(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # print("Loss time", time.time() - start_loss)
                start_update = time.time()

                loss, grads = loss_and_grad_fn(agent, pg_loss, ent_coef, entropy_loss, v_loss, vf_coef)

                # print("Update time", time.time() - start_update)

                # nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.update(agent, grads)
                mx.eval(loss, agent.parameters())

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values, b_returns
        var_y = mx.var(y_true)
        explained_var = mx.nan if var_y == 0 else 1 - mx.var(y_true - y_pred) / var_y

        # print("Global SPS:", int(global_step / (time.time() - start_time)))

        # TRY NOT TO MODIFY: record rewards for plotting purposes

    envs.close()