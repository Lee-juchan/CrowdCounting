import argparse
from collections import OrderedDict
import flwr as fl
import pytorch_lightning as pl
import albumentations as A
import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from flwr import client
from matplotlib import pyplot as plt
from model import Conv2d, MCNN
from datasets.dataset import MyDataset, aug_train, aug_val
from sklearn.model_selection import train_test_split

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


# def _get_parameters(model):
#     return [val.cpu().numpy() for _, val in model.state_dict().items()]
def _get_parameters(model):    
    parameters = [v.detach().cpu().numpy() for v in model.parameters()]
    return parameters


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
    train = [p.path for p in os.scandir('ShanghaiTech/part_B/train_data/images/')]
    valid_full = [p.path for p in os.scandir('ShanghaiTech/part_B/test_data/images/')]
    
    batch_size = 32
    epochs = 300
    max_steps = epochs * len(train) // batch_size
    
    _, valid = train_test_split(valid_full, test_size=64, random_state=42)
    
    train_loader = DataLoader(MyDataset(train, aug_train), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(MyDataset(valid, aug_val), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    ## use a small subset for validation
    len(train), len(valid)





    lr = 3e-4
    model = MCNN(lr, batch_size, max_steps)
    
    test_dataset = MyDataset(valid, aug_val)
    test_loader = DataLoader(test_dataset, batch_size=64)
    

    # 저장된 모델의 가중치를 불러옵니다.
    # model.load_state_dict(torch.load('mcnn_model_1.pth'))

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()