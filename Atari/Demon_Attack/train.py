from ES.Atari.Evolution.EvolutionStrategies import EvolutionModule
import torch
import gym
import pickle
import time
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

POPULATION = 100
MAX_BATCH_STEP = 10000

class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

if __name__ == '__main__':
    env = gym.make("DemonAttack-ram-v0")
    writer = SummaryWriter(comment="-DemonAttack-ram-v0")

    net = Net(env.observation_space.shape[0], env.action_space.n)
    print(net)
    net.apply(init_weights)
    es = EvolutionModule()
    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(POPULATION):
            noise, neg_noise = es.sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward, steps = es.eval_with_noise(env, net, noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = es.eval_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            # if (batch_steps > MAX_BATCH_STEP):
            #     break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 199:
            print("Solved in %d steps" % step_idx)
            break

        es.train_step(net, batch_noise, batch_reward, writer, step_idx)

        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))
    env.close()
    pickle.dump(net.parameters(), open("result","wb"))