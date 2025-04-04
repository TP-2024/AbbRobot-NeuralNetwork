import numpy as np
from typing import Any, List, Tuple


class PrioritizedReplay:
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.buffer: List[Any] = []
        self.priorities: List[float] = []

    def add(self, experience: Any, priority: float) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple:
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return tuple(map(list, zip(*samples)))
