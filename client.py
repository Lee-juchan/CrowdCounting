import argparse
from collections import OrderedDict
import flwr as fl
import torch
import pytorch_lightning as pl

from model import MCNN
from datasets.dataset import load_data
# from datasets.utils.logging import disable_progress_bar       내가 만든 datasets dir 아님
# disable_progress_bar()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return _get_parameters(self.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), 400, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 252, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def main() -> None:
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--node-id",
    #     type=int,
    #     choices=range(0, 10),
    #     required=True,
    #     help="Specifies the artificial data partition",
    # )
    # args = parser.parse_args()
    # node_id = args.node_id

    # Model and data
    lr = 3e-4
    model = MCNN(lr=lr)
    train_loader, val_loader, test_loader = load_data()

    # Flower client
    client = fl.client.to_client(FlowerClient(model, train_loader, val_loader, test_loader))
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()