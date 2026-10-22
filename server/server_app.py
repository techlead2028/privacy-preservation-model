from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from server.strategy import create_strategy
from model.simple_model import initial_parameters

def server_fn(context: Context):
    # Initialize the strategy with the global parameters
    strategy = create_strategy(initial_parameters())
    
    # Extract config from context or default to 3 rounds
    num_rounds = context.run_config.get("num-server-rounds", 3)
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
