"""Stable-Baselines3 feature extractors for monitor-augmented observations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _flat_dim(space: spaces.Box) -> int:
    return int(np.prod(space.shape))


class MonitorVectorExtractor(BaseFeaturesExtractor):
    """Feature extractor for vector-valued monitor encodings."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        position_hidden_dim: int = 64,
        monitor_hidden_dim: int = 64,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)

        position_dim = _flat_dim(observation_space["position"])
        monitor_dim = _flat_dim(observation_space["monitor"])

        self.position_net = nn.Sequential(
            nn.Linear(position_dim, position_hidden_dim),
            nn.ReLU(),
            nn.Linear(position_hidden_dim, position_hidden_dim),
            nn.ReLU(),
        )
        self.monitor_net = nn.Sequential(
            nn.Linear(monitor_dim, monitor_hidden_dim),
            nn.ReLU(),
            nn.Linear(monitor_hidden_dim, monitor_hidden_dim),
            nn.ReLU(),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(position_hidden_dim + monitor_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        position = observations["position"].float().view(observations["position"].shape[0], -1)
        monitor = observations["monitor"].float().view(observations["monitor"].shape[0], -1)
        position_features = self.position_net(position)
        monitor_features = self.monitor_net(monitor)
        return self.fusion_net(torch.cat([position_features, monitor_features], dim=1))


class MonitorStateEmbeddingExtractor(BaseFeaturesExtractor):
    """Feature extractor for integer monitor-state IDs."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        max_monitor_states: int = 256,
        monitor_embedding_dim: int = 16,
        position_hidden_dim: int = 64,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)

        position_dim = _flat_dim(observation_space["position"])
        self.max_monitor_states = int(max_monitor_states)

        self.position_net = nn.Sequential(
            nn.Linear(position_dim, position_hidden_dim),
            nn.ReLU(),
            nn.Linear(position_hidden_dim, position_hidden_dim),
            nn.ReLU(),
        )
        self.monitor_embedding = nn.Embedding(self.max_monitor_states, monitor_embedding_dim)
        self.fusion_net = nn.Sequential(
            nn.Linear(position_hidden_dim + monitor_embedding_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        position = observations["position"].float().view(observations["position"].shape[0], -1)
        monitor = observations["monitor"].view(observations["monitor"].shape[0], -1)
        monitor_index = monitor[:, 0].round().long().clamp(min=0, max=self.max_monitor_states - 1)
        position_features = self.position_net(position)
        monitor_features = self.monitor_embedding(monitor_index)
        return self.fusion_net(torch.cat([position_features, monitor_features], dim=1))
