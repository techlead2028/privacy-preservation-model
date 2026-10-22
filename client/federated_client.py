import numpy as np
import flwr as fl
from model.simple_model import initial_parameters
from client.trainer import ClientTrainer

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, trainer: ClientTrainer):
        self.trainer = trainer

    def get_parameters(self, config):
        return initial_parameters()

    def fit(self, parameters, config):
        # 1. Delegate local training to the trainer interface
        updated_parameters, num_examples, metrics = self.trainer.train(parameters, config)

        # 2. Compute the exact delta (local - global)
        delta = [u - p for u, p in zip(updated_parameters, parameters)]

        # 3. Future Member 3 privacy processing on delta can happen here
        
        return delta, num_examples, metrics

    def evaluate(self, parameters, config):
        return self.trainer.evaluate(parameters, config)