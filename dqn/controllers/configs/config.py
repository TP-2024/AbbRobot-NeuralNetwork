from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    episodes: int = Field(..., description="Number of training episodes")
    max_steps: int = Field(..., description="Maximum number of steps per episode")
    learning_rate: float = Field(..., description="Learning rate for the optimizer")
    gamma: float = Field(..., description="Discount factor for future rewards")


class DQNConfig(BaseConfig):
    batch_size: int = Field(..., description="Batch size for sampling from the replay buffer")
    epsilon_start: float = Field(..., description="Starting epsilon value for epsilon-greedy action selection")
    epsilon_end: float = Field(..., description="Minimum epsilon value")
    epsilon_decay: float = Field(..., description="Decay factor for epsilon per episode")
    memory_size: int = Field(..., description="Capacity of the replay buffer")
    update_target_every: int = Field(..., description="Frequency (in episodes) to update the target network")
    gradient_clip: float = Field(..., description="Maximum gradient clipping value")


class PGConfig(BaseConfig):
    pass


class ActorCriticConfig(BaseConfig):
    value_loss_coef: float = Field(..., description="Coefficient weighting the critic (value) loss")
