import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

class FederatedCoordinator(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: List[np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_parameters = initial_parameters
        # We need to tell the server what the initial parameters are
        self.initial_parameters = ndarrays_to_parameters(initial_parameters)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average of deltas."""
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results to ndarrays (these are deltas) and weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Calculate weighted delta: Σ (num_examples_i / total_examples) * delta_i
        weighted_delta = aggregate(weights_results)

        # Update global parameters: global_parameters = global_parameters + weighted_delta
        self.global_parameters = [
            global_param + delta_param 
            for global_param, delta_param in zip(self.global_parameters, weighted_delta)
        ]

        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(self.global_parameters)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("WARNING : No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

def create_strategy(initial_parameters):
    return FederatedCoordinator(initial_parameters=initial_parameters)