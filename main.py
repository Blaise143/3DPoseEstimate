import torch
import torch.nn as nn
import os
import json
import random
from utils import plot_keypoints  # , load_checkpoint
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datasets import TestCustomDataset, MocapDataset
from models.priors import VariationalAutoEncoder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import ConcatDataset


wandb_logger = WandbLogger(save_dir="../checkpoints",
                           project="Mirror_Project",)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="val_loss",
    # filename="vae_model_{epoch:02d}_val_loss_{val_loss:.2f}",
    filename="mocap_model_again",
    verbose=False,
    mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="train_reconstruction_loss",
    mode="min",
    patience=10
)
path = "data/zju-m-seq1/annots/3"
path_2 = "data/zju-m-seq1/annots/6"

dataset_1 = TestCustomDataset(path)
dataset_2 = TestCustomDataset(path_2)
custom_data = ConcatDataset([dataset_1, dataset_2])

mocap_data = MocapDataset(path="data/HuMiD-yukagawa-clips")
# print(f"mocap len: {len(mocap_data)}")
# exit()

first_img = mocap_data[0][0][0]
sec_img = mocap_data[1][0][0]
print(first_img)
print(sec_img)
# exit()


layers_order = [38, 35, 30, 25, 20, 15]
latent_dim = 15


def run_prior():
    vae_model = VariationalAutoEncoder(
        layers_order=layers_order, latent_dim=latent_dim, dataset=mocap_data, learning_rate=1e-4)

    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=50,
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         logger=wandb_logger,
                         log_every_n_steps=3)  # ,
    # check_val_every_n_epoch=20)
    trainer.fit(vae_model)


# run_prior()
# print("EXITING!")
# exit()


def load_checkpoint(checkpoint_path: str, model):
    checkpoint_state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(checkpoint_state_dict)
    return model


inference_model = VariationalAutoEncoder(
    layers_order=layers_order, latent_dim=latent_dim)


model = load_checkpoint("checkpoints/mocap_model_again.ckpt", inference_model)
print("printing model")
print(model)
model.eval()
random_tensor = torch.randn(15)

out_passthrough_1 = model(first_img.unsqueeze(0))[
    0][0].squeeze(0).detach().view(-1, 2)
out_passthrough_2 = model(sec_img.unsqueeze(0))[
    0][0].squeeze(0).detach().view(-1, 2)
print("out 1")
print(out_passthrough_1)
plot_keypoints(out_passthrough_1, title="mocap_reconstructed_1_again")
plt.savefig("output_images/passthrough/img_1_pred_again.png")
print("out 2")
print(out_passthrough_2)
plot_keypoints(out_passthrough_2, title="mocap_reconstructed_2_again")
plt.savefig("output_images/passthrough/img_2_pred_again.png")
plot_keypoints(first_img.view(-1, 2), title="actual_1")
plt.savefig("output_images/passthrough/img_1.png")
plot_keypoints(sec_img.view(-1, 2), title="actual_2")
plt.savefig("output_images/passthrough/img_2.png")

# exit()

decoder = model.decoder
print("printing decoder")
print(decoder)
out = decoder(random_tensor).view(-1, 2).detach()
plot_keypoints(out, title="random plot")
plt.savefig("output_images/random_generations/mocap_random_plot_again.png")
plt.show()
print(out)
exit()


def predict_and_plot(checkpoint_path: str = "checkpoints/mocap_model_latest.ckpt"):
    # checkpoint_path = ""

    model = load_checkpoint(checkpoint_path=checkpoint_path,
                            model=inference_model)
    random_indices = random.sample(range(len(mocap_data)), 2)
    random_keypoints = [mocap_data[idx][0] for idx in random_indices]

    model.eval()

    for point in random_keypoints:
        real, virtual = point[0], point[1]
        real_pred, virtual_pred = model(
            real.unsqueeze(0)), model(virtual.unsqueeze(0))

        real, virtual = real.view(-1, 2), virtual.view(-1, 2)
        real_pred, virtual_pred = real_pred[0].squeeze(
            0).view(-1, 2), virtual_pred[0].squeeze(0).view(-1, 2)

        virtual_plot = plot_keypoints(virtual, title="actual virtual")
        plt.savefig("output_images/passthrough/actual_virtual.png")
        v_pred_plot = plot_keypoints(virtual_pred, title="predicted virtual")
        plt.savefig("output_images/passthrough/pred_virtual.png")
        real_plot = plot_keypoints(real, title="Actual Real")
        plt.savefig("output_images/passthrough/actual_real.png")
        r_pred_plot = plot_keypoints(real_pred, title="predicted real")
        plt.savefig("output_images/passthrough/predicted_real.png")
        # plt.savefig(real_plot, "real.png")
        # plt.savefig(v_plot, "vitual.png")
        plt.show()


predict_and_plot()
# predict_and_plot()


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
