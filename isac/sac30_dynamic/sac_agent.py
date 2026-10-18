from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, output_activation=nn.Identity):
    layers = []
    for index in range(len(sizes) - 1):
        activation = nn.ReLU if index < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[index], sizes[index + 1]), activation()])
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, width=256):
        super().__init__()
        self.body = mlp([state_dim, width, width], nn.ReLU)
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)

    def forward(self, state, deterministic=False, log_probability=True):
        hidden = self.body(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-10.0, 1.0)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        logp = None
        if log_probability:
            logp = distribution.log_prob(raw).sum(-1, keepdim=True)
            logp -= (2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))).sum(-1, keepdim=True)
        return action, logp


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, width=256):
        super().__init__()
        self.q1 = mlp([state_dim + action_dim, width, width, 1])
        self.q2 = mlp([state_dim + action_dim, width, width, 1])

    def forward(self, state, action):
        joined = torch.cat([state, action], dim=-1)
        return self.q1(joined), self.q2(joined)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100_000, rng=None):
        self.capacity = int(capacity)
        self.pointer = 0
        self.size = 0
        self.state = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.action = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_state = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)
        # ``None`` preserves the legacy use of NumPy's global RNG.  Controlled
        # experiments can inject a Generator so replay sampling cannot perturb
        # the evolutionary optimizer's global random stream.
        self.rng = rng

    def add(self, state, action, reward, next_state, done):
        i = self.pointer
        self.state[i], self.action[i] = state, action
        self.reward[i], self.next_state[i], self.done[i] = reward, next_state, done
        self.pointer = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = (np.random.randint(0, self.size, size=batch_size)
                   if self.rng is None
                   else self.rng.integers(0, self.size, size=batch_size))
        convert = lambda value: torch.as_tensor(value[indices], device=device)
        return convert(self.state), convert(self.action), convert(self.reward), convert(self.next_state), convert(self.done)

    def state_dict(self):
        count = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "pointer": self.pointer,
            "size": self.size,
            "state": self.state[:count].copy(),
            "action": self.action[:count].copy(),
            "reward": self.reward[:count].copy(),
            "next_state": self.next_state[:count].copy(),
            "done": self.done[:count].copy(),
            "rng_state": (None if self.rng is None else copy.deepcopy(self.rng.bit_generator.state)),
        }

    def load_state_dict(self, data):
        if int(data["capacity"]) != self.capacity:
            raise ValueError("Replay-buffer capacity does not match checkpoint.")
        count = self.capacity if int(data["size"]) == self.capacity else int(data["size"])
        for name in ["state", "action", "reward", "next_state", "done"]:
            target = getattr(self, name)
            value = np.asarray(data[name], dtype=np.float32)
            if value.shape != target[:count].shape:
                raise ValueError(f"Replay-buffer {name} shape does not match checkpoint.")
            target[:count] = value
        self.pointer, self.size = int(data["pointer"]), int(data["size"])
        if data.get("rng_state") is not None:
            if self.rng is None:
                self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(data["rng_state"])


class SACAgent:
    def __init__(self, state_dim, action_dim, device="cpu", width=256, gamma=0.99,
                 learning_rate=3e-4, batch_size=256, capacity=100_000, alpha=0.2,
                 target_entropy=None, replay_rng=None):
        self.device = torch.device(device)
        self.gamma, self.batch_size, self.tau = gamma, batch_size, 0.005
        self.actor = Actor(state_dim, action_dim, width).to(self.device)
        self.critic = Critic(state_dim, action_dim, width).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = (-float(action_dim) if target_entropy is None
                               else float(target_entropy))
        self.replay = ReplayBuffer(state_dim, action_dim, capacity, replay_rng)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state_tensor, deterministic, False)
        return action[0].cpu().numpy()

    def q_bounds(self, state, action):
        """Return pessimistic/optimistic twin-critic estimates for one pair."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q1, q2 = self.critic(state_tensor, action_tensor)
        values = torch.stack([q1, q2])
        return float(values.min().cpu()), float(values.max().cpu())

    def update(self, updates=1):
        if self.replay.size < self.batch_size:
            return
        for _ in range(updates):
            state, action, reward, next_state, done = self.replay.sample(self.batch_size, self.device)
            with torch.no_grad():
                next_action, next_logp = self.actor(next_state)
                q1_target, q2_target = self.target(next_state, next_action)
                target = reward + (1.0 - done) * self.gamma * (torch.minimum(q1_target, q2_target) - self.alpha * next_logp)
            q1, q2 = self.critic(state, action)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            sampled_action, logp = self.actor(state)
            aq1, aq2 = self.critic(state, sampled_action)
            actor_loss = (self.alpha.detach() * logp - torch.minimum(aq1, aq2)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
            with torch.no_grad():
                for current, target_parameter in zip(self.critic.parameters(), self.target.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(self.tau * current)

    def initialize_actor_mean(self, action, log_std=-3.0):
        clipped = np.clip(np.asarray(action), -0.999, 0.999)
        with torch.no_grad():
            self.actor.mean.weight.zero_()
            self.actor.mean.bias.copy_(torch.as_tensor(np.arctanh(clipped), dtype=torch.float32, device=self.device))
            self.actor.log_std.weight.zero_()
            self.actor.log_std.bias.fill_(float(log_std))

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "target": self.target.state_dict(), "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.target.load_state_dict(data.get("target", data["critic"]))
        with torch.no_grad():
            self.log_alpha.copy_(data["log_alpha"].to(self.device))

    def training_state_dict(self):
        """Return a complete episode-boundary state for exact resumption."""
        result = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            result["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        return result

    def load_training_state_dict(self, data):
        """Restore a state produced by :meth:`training_state_dict`."""
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.target.load_state_dict(data["target"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(data["log_alpha"].to(self.device))
        self.replay.load_state_dict(data["replay"])
        torch.set_rng_state(data["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_states" in data:
            torch.cuda.set_rng_state_all(data["cuda_rng_states"])
