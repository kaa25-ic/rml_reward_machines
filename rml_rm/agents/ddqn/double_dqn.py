"""Stable-Baselines3-compatible Double DQN."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class DoubleDQN(DQN):
    """DQN with Double-DQN target action selection."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                next_online_q_values = self.q_net(replay_data.next_observations)
                next_actions = next_online_q_values.argmax(dim=1, keepdim=True)
                next_target_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_target_q_values, dim=1, index=next_actions)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
