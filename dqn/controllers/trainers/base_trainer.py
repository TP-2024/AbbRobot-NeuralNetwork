from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
