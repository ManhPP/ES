import numpy as np
import torch
import time


class EvolutionModule:

    def __init__(
            self,
            sigma = 0.1,
            learning_rate = 0.001,
            decay = 1.0,
    ):
        np.random.seed(int(time.time()))
        self.SIGMA = float(sigma)
        self.LEARNING_RATE = float(sigma)
        self.decay =float(decay)

    def sample_noise(self, net):
        pos, neg = [], []
        for p in net.parameters():
            noise = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
            pos.append(noise)
            neg.append(-noise)
        return pos, neg

    def evaluate(self, env, net):
        obs = env.reset()
        reward = 0.0
        steps = 0
        while (True):
            obs_v = torch.Tensor([obs])
            act_prob = net(obs_v)
            acts = act_prob.max(dim=1)[1]
            obs, r, done, _ = env.step(acts.data.numpy()[0])
            reward += r
            steps += 1
            if done:
                break
        return reward, steps

    def eval_with_noise(self, env, net, noise):
        old_params = net.state_dict()
        for p, p_n in zip(net.parameters(), noise):
            p.data += self.SIGMA * p_n
        r, s = self.evaluate(env, net)
        net.load_state_dict(old_params)
        return r, s


    def compute_ranks(self, x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size)
        y -= .5
        return y

    def train_step(self, net, batch_noise, batch_reward, writer, step_idx):
        norm_reward = np.array(batch_reward)
        norm_reward -= np.mean(norm_reward)
        std = np.std(norm_reward)
        if abs(std) > 1e-6:
            norm_reward /= std

        batch_rank = self.compute_centered_ranks(np.array(batch_reward))
        weighted_noise = None
        for noise, reward in zip(batch_noise, batch_rank):
            if weighted_noise is None:
                weighted_noise = [reward * p_n for p_n in noise]
            else:
                for w_n, p_n in zip(weighted_noise, noise):
                    w_n += reward * p_n

        m_updates = []
        for p, p_update in zip(net.parameters(), weighted_noise):
            update = p_update / (len(batch_reward) * self.SIGMA)
            p.data += self.LEARNING_RATE * update
            m_updates.append(torch.norm(update))

        # self.LEARNING_RATE *= self.decay
        # self.SIGMA *= self.sigma_decay
        writer.add_scalar("update", np.mean(m_updates), step_idx)
