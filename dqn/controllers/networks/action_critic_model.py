import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Shared common network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        # Actor head: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head: outputs state value estimate
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
