import torch
import torch.nn as nn
import os
import json
import random
from utils import plot_keypoints
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dataloaders import TestCustomDataset
from models.priors import VariationalAutoEncoder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(save_dir="../checkpoints",
                           project="Mirror_Project",)

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints",
    monitor="val_loss",
    filename="vae_model_{epoch:02d}_val_loss_{val_loss:.2f}",
    verbose=False,
    mode="min"
)
path = "../data/zju-m-seq1/annots/3"
custom_data = TestCustomDataset(path)

# print(custom_data[0][0])
# first = custom_data[0][0]
# print(first)
# plot = plot_keypoints(first.view(-1, 2))
# plt.show()
layers_order = [38, 35, 32, 30, 25, 18, 20, 15]
latent_dim = 10


def run_prior():
    vae_model = VariationalAutoEncoder(
        layers_order=layers_order, latent_dim=latent_dim, dataset=custom_data)

    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=5000,
                         callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         log_every_n_steps=30)
    trainer.fit(vae_model)


# run_prior()


def load_checkpoint(checkpoint_path: str):
    checkpoint_state_dict = torch.load(checkpoint_path)["state_dict"]
    # print(checkpoint_state_dict)
    model = VariationalAutoEncoder(
        layers_order=layers_order,
        latent_dim=latent_dim,
        dataset=custom_data
    )
    model.load_state_dict(checkpoint_state_dict)
    # print(model)
    return model


def predict_and_plot(checkpoint_path: str = "../checkpoints/vae_model_epoch=3675_val_loss_val_loss=294.52.ckpt"):
    # checkpoint_path = ""
    model = load_checkpoint(checkpoint_path=checkpoint_path)
    # print(model)

    random_indices = random.sample(range(len(custom_data)), 1)
    random_keypoints = [custom_data[idx] for idx in random_indices]

    # print(random_indices)
    # print(random_keypoints)
    # print(custom_data[0])
    model.eval()
    for point in random_keypoints:
        real, virtual = point[0], point[1]
        real_pred, virtual_pred = model(
            real.unsqueeze(0)), model(virtual.unsqueeze(0))

        real, virtual = real.view(-1, 2), virtual.view(-1, 2)
        real_pred, virtual_pred = real_pred[0].squeeze(
            0).view(-1, 2), virtual_pred[0].squeeze(0).view(-1, 2)

        virtual_plot = plot_keypoints(virtual, title="actual virtual")
        v_pred_plot = plot_keypoints(virtual_pred, title="predicted virtual")
        real_plot = plot_keypoints(real, title="Actual Real")
        r_pred_plot = plot_keypoints(real_pred, title="predicted real")
        plt.show()


predict_and_plot(
    checkpoint_path="../checkpoints/vae_model_epoch=2362_val_loss_val_loss=286.11.ckpt")
# predict_and_plot()
