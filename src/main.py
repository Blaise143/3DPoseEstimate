import torch
import torch.nn as nn
import os
import json
from utils import plot_keypoints
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dataloaders import TestCustomDataset
from models.priors import VariationalAutoEncoder
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(save_dir="../", project="Mirror_Project")

path = "../../data/zju-m-seq1/annots/3"
custom_data = TestCustomDataset(path)
# print(custom_data[0][0])
# first = custom_data[0][0]
# print(first)
# plot = plot_keypoints(first.view(-1, 2))
# plt.show()
layers_order = [38, 35, 30, 25, 20, 15]
latent_dim = 10


def run_prior():
    vae_model = VariationalAutoEncoder(
        layers_order=layers_order, latent_dim=latent_dim, dataset=custom_data)

    trainer = pl.Trainer(accelerator="gpu", devices="auto",
                         max_epochs=1000, logger=wandb_logger, log_every_n_steps=10)
    trainer.fit(vae_model)


run_prior()
