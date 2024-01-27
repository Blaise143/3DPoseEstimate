import torch
import torch.nn as nn
import os
import json
import random
from utils import plot_keypoints  # , load_checkpoint
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dataloaders import TestCustomDataset
from models.priors import VariationalAutoEncoder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import ConcatDataset

wandb_logger = WandbLogger(save_dir="../checkpoints",
                           project="Mirror_Project",)

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints",
    monitor="val_loss",
    # filename="vae_model_{epoch:02d}_val_loss_{val_loss:.2f}",
    filename="concat_model_2",
    verbose=False,
    mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="train_reconstruction_loss",
    mode="min",
    patience=10

)
path = "../data/zju-m-seq1/annots/3"
path_2 = "../data/zju-m-seq1/annots/6"

dataset_1 = TestCustomDataset(path)
dataset_2 = TestCustomDataset(path_2)
custom_data = ConcatDataset([dataset_1, dataset_2])
# print(custom_data[0][0])
# first = custom_data[0][0]
# print(first)
# plot = plot_keypoints(first.view(-1, 2))
# plt.show()
layers_order = [38, 30, 20]
latent_dim = 15


def run_prior():
    vae_model = VariationalAutoEncoder(
        layers_order=layers_order, latent_dim=latent_dim, dataset=custom_data, learning_rate=1e-4)

    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=10_000,
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         logger=wandb_logger,
                         log_every_n_steps=3)  # ,
    # check_val_every_n_epoch=20)
    trainer.fit(vae_model)


# run_prior()
# exit()

def load_checkpoint(checkpoint_path: str, model):
    checkpoint_state_dict = torch.load(checkpoint_path)["state_dict"]
    # print(checkpoint_state_dict)
    # model = VariationalAutoEncoder(
    #     layers_order=layers_order,
    #     latent_dim=latent_dim,
    #     dataset=custom_data
    # )
    model.load_state_dict(checkpoint_state_dict)
    # print(model)
    return model


inference_model = VariationalAutoEncoder(
    layers_order=layers_order, latent_dim=latent_dim, dataset=custom_data)


def predict_and_plot(checkpoint_path: str = "../checkpoints/vae_model_epoch=8861_val_loss_val_loss=0.02.ckpt"):
    # checkpoint_path = ""

    model = load_checkpoint(checkpoint_path=checkpoint_path,
                            model=inference_model)
    random_indices = random.sample(range(len(custom_data)), )
    random_keypoints = [custom_data[idx] for idx in random_indices]

    model.eval()

    for point in random_keypoints:
        real, virtual = point[0], point[1]
        real_pred, virtual_pred = model(
            real.unsqueeze(0)), model(virtual.unsqueeze(0))

        real, virtual = real.view(-1, 2), virtual.view(-1, 2)
        real_pred, virtual_pred = real_pred[0].squeeze(
            0).view(-1, 2), virtual_pred[0].squeeze(0).view(-1, 2)

        virtual_plot = plot_keypoints(virtual, title="actual virtual")
        plt.savefig("../output_images/passthrough/actual_virtual.png")
        v_pred_plot = plot_keypoints(virtual_pred, title="predicted virtual")
        plt.savefig("../output_images/passthrough/pred_virtual.png")
        real_plot = plot_keypoints(real, title="Actual Real")
        plt.savefig("../output_images/passthrough/actual_real.png")
        r_pred_plot = plot_keypoints(real_pred, title="predicted real")
        plt.savefig("../output_images/passthrough/predicted_real.png")
        # plt.savefig(real_plot, "real.png")
        # plt.savefig(v_plot, "vitual.png")
        plt.show()


predict_and_plot(checkpoint_path="../checkpoints/concat_model_2.ckpt")
# predict_and_plot()
#
exit()
inference_model = VariationalAutoEncoder(
    layers_order=layers_order, latent_dim=latent_dim, dataset=custom_data)
checkpoint_path = "../checkpoints/concat_model_2.ckpt"
state_dict = torch.load(checkpoint_path)["state_dict"]
# print(state_dict)
# print(inference_model.decoder)
inference_model.load_state_dict(state_dict=state_dict)

decoder = inference_model.decoder

decoder.eval()


def generate(n: int):
    for i in range(n):
        rand_tensor = torch.randn(15)
        out = decoder(rand_tensor).view(-1, 2)
        plot_keypoints(out, title="random_plot")
        plt.savefig(
            "../output_images/random_generations/concat_gen_fewer_dim_" + str(i)+".png")
        plt.show()

# def infer(x: torch.tensor):


# print(inference_model)
# random_tensor = torch.randn(10)

# print(random_tensor)
# print(decoder)

# out = decoder(random_tensor).view(-1, 2)
# plot_keypoints(out, title="random noise plot")
# plt.savefig("../output_images/noise_plot_shuffled.png")
# inference_model.setup()
# inference_model_loader = inference_model.train_dataloader()
# data_list = []
# for data in inference_model_loader:
    # data_list.extend(data)
# all_data_tensor = torch.stack(data_list)
# print(all_data_tensor)
# print(len(data_list))
# training = []
# for i in data_list:
    # training.extend(i)
# print(training)
# training = torch.stack(training)
# print(training.shape)
# mean_training = torch.mean(training, dim=0)
# new_train = training/mean_training
# print(mean_training.shape)
# mean_training = mean_training.view(-1, 2)
# print(new_train.shape)
# print(mean_training)
# plot_keypoints(mean_training, title="mean plot")
# plt.savefig("../output_images/reduced_model_.png")
# plt.show()
generate(5)
