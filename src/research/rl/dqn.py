"""DQN agent for the exploration MDP (spec §14.3).

A deliberately small Deep Q-Network: the state is 5-dim and there are 5
discrete actions, so a 2×64 MLP is plenty. Standard ingredients — replay
buffer, target network with soft (Polyak) updates, ε-greedy exploration,
Huber loss. Pure PyTorch, no RL framework, fully seedable for reproducible
tests.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .env import N_ACTIONS, STATE_DIM


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.95
    buffer_size: int = 50_000
    batch_size: int = 64
    warmup: int = 500  # steps of random play before learning
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 4_000
    target_tau: float = 0.01  # soft target update rate
    seed: int = 0


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self._buf.append((s.astype(np.float32), a, r, s2.astype(np.float32), done))

    def sample(self, batch_size: int, rng: random.Random) -> tuple[torch.Tensor, ...]:
        batch = rng.sample(self._buf, batch_size)
        s, a, r, s2, done = zip(*batch, strict=True)
        return (
            torch.from_numpy(np.stack(s)),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.from_numpy(np.stack(s2)),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    """ε-greedy DQN with a soft-updated target network."""

    def __init__(self, cfg: DQNConfig | None = None) -> None:
        self.cfg = cfg or DQNConfig()
        torch.manual_seed(self.cfg.seed)
        self._py_rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)
        self.q = QNetwork(hidden=self.cfg.hidden)
        self.target = QNetwork(hidden=self.cfg.hidden)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self._learn_steps = 0

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        if self._np_rng.random() < eps:
            return int(self._np_rng.integers(0, N_ACTIONS))
        with torch.no_grad():
            q = self.q(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.push(s, a, r, s2, done)

    def learn(self) -> float | None:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup):
            return None
        s, a, r, s2, done = self.buffer.sample(self.cfg.batch_size, self._py_rng)
        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(s2).max(dim=1, keepdim=True).values
            target = r + self.cfg.gamma * next_q * (1.0 - done)
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self._soft_update()
        self._learn_steps += 1
        return float(loss.item())

    def _soft_update(self) -> None:
        tau = self.cfg.target_tau
        for tp, p in zip(self.target.parameters(), self.q.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.q.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str) -> DQNAgent:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(DQNConfig(**blob["cfg"]))
        agent.q.load_state_dict(blob["state_dict"])
        agent.target.load_state_dict(blob["state_dict"])
        return agent
