from flwr.client import ClientApp
from flwr.common import Context
from client.federated_client import FederatedClient
from client.trainer import ToyTrainer

def client_fn(context: Context):
    # Instantiate the appropriate trainer and inject into client
    trainer = ToyTrainer()
    return FederatedClient(trainer=trainer).to_client()

app = ClientApp(client_fn=client_fn)
