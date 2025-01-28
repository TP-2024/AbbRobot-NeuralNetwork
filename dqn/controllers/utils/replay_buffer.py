import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []

    def add(self, experience, priority):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return zip(*samples)
