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
from models.priors import VQVAE

wandb_logger = WandbLogger(save_dir="../checkpoints",
                           project="Mirror_Project",)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="val_recon_loss",
    # filename="vae_model_{epoch:02d}_val_loss_{val_loss:.2f}",
    filename="ema",
    verbose=False,
    mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="val_recon_loss",
    mode="min",
    patience=10
)
path = "data/zju-m-seq1/annots/3"
path_2 = "data/zju-m-seq1/annots/6"

dataset_1 = TestCustomDataset(path)
dataset_2 = TestCustomDataset(path_2)
custom_data = ConcatDataset([dataset_1, dataset_2])

print(len(custom_data))
# exit()


mocap_data = MocapDataset(path="data/HuMiD-yukagawa-clips")
print(f"mocap len: {len(mocap_data)}")
# exit()

# layers_order = [38, 70, 100, 128]
# latent_dim = 128
layers_order = [38, 35, 30, 25, 20, 15]
latent_dim = 15

def run_prior():
    vae_model = VariationalAutoEncoder(
        layers_order=layers_order, latent_dim=latent_dim, dataset=mocap_data, learning_rate=1e-4)

    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=15,
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback],
                         logger=wandb_logger,
                         log_every_n_steps=3)  # ,
    # check_val_every_n_epoch=20)
    trainer.fit(vae_model)


def run_vq_prior():
    model = VQVAE(
        encoder_layers=layers_order,
        decoder_layers=list(reversed(layers_order)),
        num_embeddings=128,
        embedding_dim=latent_dim,
        commitment_cost=0.3,
        learning_rate=1e-4,
        dataset=mocap_data,
        denoise=False,
        use_ema=True
    )
    # print(model)
    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=30,
                         callbacks=[checkpoint_callback],
                         # early_stopping_callback],
                         logger=wandb_logger,
                         log_every_n_steps=3)  # ,
    # check_val_every_n_epoch=20)
    trainer.fit(model)


run_vq_prior()
print("EXITING")
exit()
# run_prior()
# print("EXITING!")
# exit()
vq_model = VQVAE(
    encoder_layers=layers_order,
    decoder_layers=list(reversed(layers_order)),
    num_embeddings=128,
    embedding_dim=latent_dim,
    commitment_cost=0.3,
    learning_rate=1e-4,
    dataset=mocap_data,
    denoise=False,
    use_ema=True
)


def load_checkpoint(checkpoint_path: str, model):
    checkpoint_state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(checkpoint_state_dict)
    return model


vq_model = load_checkpoint("./checkpoints/ema.ckpt", vq_model)

print(vq_model)
# exit()
codebook = vq_model.quantizer.embedding.weight.data
print(f"code book: {codebook.shape}")
decoder = vq_model.decoder
for i in range(10):#(len(codebook)-490):
    # print(len(codebook))
    vector = codebook[i]
    out = decoder(vector.unsqueeze(0)).detach().squeeze(0).view(-1, 2)
    # print(out)
    plot_keypoints(out, title=f"Vector {i}")
    plt.savefig(f"output_images/vq3/codebook/{i}_ema.png")
    # print(out.shape)
exit()


def load_data(idx: int, data=mocap_data):
    (real, _), (virtual, _) = mocap_data[idx]
    return real, virtual


vq_model.eval()
decoder = vq_model.decoder
print(decoder)
random_tensor = torch.randn(15)
# out = decoder(random_tensor).view(-1, 2)
# print(out)
# (real, _), (virtual, _) = mocap_data[0]
real, virtual = load_data(15)
# print(f"real shape: {real.shape}, virtual shape: {virtual.shape}")
# print("___")
print(real.unsqueeze(0).shape)
real_pred, _ = vq_model(real.unsqueeze(0))
virtual_pred, _ = vq_model(virtual.unsqueeze(0))

print(real_pred.squeeze(0))
real_pred, virtual_pred = real_pred.squeeze(0).detach(
).view(-1, 2), virtual_pred.squeeze(0).detach().view(-1, 2)
print(
    f"real pred shape: {real_pred.shape}, virtual pred shape: {virtual_pred.shape}")

plot_keypoints(real.view(-1, 2), title="Real input")
# plt.show()
# plt.savefig("output_images/vq/passthrough/four.png")
plot_keypoints(real_pred, title="Predicted Real")
# plt.savefig("output_images/vq/passthrough/four_pred.png")
# plt.show()
plot_keypoints(virtual.view(-1, 2), title="Virtual Input")
# plt.savefig("output_images/vq_outputs/real_two.png")
# plt.savefig("output_images/vq/passthrough/four_v.png")

# plt.show()
plot_keypoints(virtual_pred, title="virtual pred")
# plt.savefig("output_images/vq/passthrough/four_v_pred.png")

# plt.show()
# print(f"real_pred shape: {real_pred.shape}")
# exit()

# inference_model = VariationalAutoEncoder(
# layers_order=layers_order, latent_dim=latent_dim)


# model = load_checkpoint("checkpoints/mocap_model_again.ckpt", inference_model)
# print("printing model")
# print(model)
# model.eval()
# random_tensor = torch.randn(15)

# out_passthrough_1 = model(first_img.unsqueeze(0))[
#     0][0].squeeze(0).detach().view(-1, 2)
# out_passthrough_2 = model(sec_img.unsqueeze(0))[
#     0][0].squeeze(0).detach().view(-1, 2)
# print("out 1")
# print(out_passthrough_1)
# plot_keypoints(out_passthrough_1, title="mocap_reconstructed_1_again")
# plt.savefig("output_images/passthrough/img_1_pred_again.png")
# print("out 2")
# print(out_passthrough_2)
# plot_keypoints(out_passthrough_2, title="mocap_reconstructed_2_again")
# plt.savefig("output_images/passthrough/img_2_pred_again.png")
# plot_keypoints(first_img.view(-1, 2), title="actual_1")
# plt.savefig("output_images/passthrough/img_1.png")
# plot_keypoints(sec_img.view(-1, 2), title="actual_2")
# plt.savefig("output_images/passthrough/img_2.png")

# exit()

# decoder = model.decoder
decoder = vq_model.decoder
print("printing decoder")
print(decoder)
random_tensor = torch.randn(15)
out = decoder(random_tensor).view(-1, 2).detach()
# plot_keypoints(out, title="random plot")
# plt.savefig("output_images/vq/generations/three.png")
# plt.show()
# print(out)
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
