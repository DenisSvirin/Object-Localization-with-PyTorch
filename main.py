from Get_XML_Data import xml_to_csv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from LocDataset import LocDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.models.mobilenetv3 import (
mobilenet_v3_small,
MobileNet_V3_Small_Weights,
)
import numpy as np
import os
from IPython.display import clear_output
import torch.optim as optim
import torch
import torch.nn as nn

def train_epoch(model, optimizer, criterion_class, criterion_box):
    """
    Training cycle for 1 epoch
    """
    loss_log, acc_log = [], []
    model.train()
    for x_batch, y_batch, box_batch in train_loader:
        x_batch, y_batch, box_batch = (
            x_batch.to(device),
            y_batch.to(device),
            box_batch.to(device),
        )

        optimizer.zero_grad()

        pred_class, pred_box = model(x_batch)

        loss_class = criterion_class(pred_class, y_batch)
        loss_box = criterion_box(pred_box, box_batch)


        pred_class = pred_class.cpu().detach().numpy()

        acc_log.append(
            accuracy_score(np.argmax(pred_class, axis=1), y_batch.cpu().numpy())
        )
        loss_log.append((loss_class + loss_box).item())

        (loss_class + loss_box).backward()
        optimizer.step()

    return loss_log, acc_log


@torch.no_grad()
def test_epoch(model, criterion_class, criterion_box):
    """
    Testing cycle for 1 epoch
    """
    loss_log, acc_log = [], []

    model.eval()
    for x_batch, y_batch, box_batch in train_loader:
        x_batch, y_batch, box_batch = (
            x_batch.to(device),
            y_batch.to(device),
            box_batch.to(device),
        )

        pred_class, pred_box = model(x_batch)
        loss_class = criterion_class(pred_class, y_batch)
        loss_box = criterion_box(pred_box, box_batch)

        pred_class = pred_class.cpu().numpy()

        acc_log.append(
            accuracy_score(np.argmax(pred_class, axis=1), y_batch.cpu().numpy())
        )
        loss_log.append((loss_class + loss_box).item())

    return loss_log, acc_log


def train(model, optimizer, criterion_class, criterion_box, epochs):
    """
    Main train cycle
    """
    for epoch in tqdm(range(epochs)):
        train_loss, acc_train = train_epoch(
            model, optimizer, criterion_class, criterion_box
        )
        test_loss, acc_test = test_epoch(model, criterion_class, criterion_box)
        print()
        print(
            "train_loss: ",
            np.mean(train_loss),
            " | ",
            "test_loss: ",
            np.mean(test_loss),
        )
        print("test acc:", np.mean(acc_train), "|", "test acc:", np.mean(acc_test))


class LocCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trained_model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        for param in self.trained_model.parameters():
            param.requires_grad = False
        self.trained_layers = list(self.trained_model.children())[:-1]
        self.body = nn.Sequential(*self.trained_layers)

        self.class_head = nn.Sequential(
            nn.Linear(576, 240),
            nn.ELU(),
            nn.Linear(240, 120),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(120, 2),
        )
        self.box_head = nn.Sequential(
            nn.Linear(576, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.Linear(300, 4),
        )

    def forward(self, X):
        out = self.get_embedding(X)

        out_class = self.class_head(out)
        out_box = self.box_head(out)

        return out_class, out_box

    def get_embedding(self, X):
        out = self.body(X)
        return out.flatten(1, -1)

if __name__ == "__main__":
    image_directory = "dataset/images"
    info_directory = "dataset/annot"

    # extracting info from xml files
    classes = ['dog', 'cat']
    df = xml_to_csv(image_directory, info_directory)

    # Creating train/test data loaders
    traindf, testdf = train_test_split(df, test_size=0.25)
    train_dataset = LocDataset(traindf)
    test_dataset = LocDataset(testdf)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # Selecting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initializing model, optimizer and losses
    model = LocCNN()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion_class = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()
    train(model, optimizer, criterion_class, criterion_box, 20)
