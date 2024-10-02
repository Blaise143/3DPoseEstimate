import torch
import pytorch_lightning as pl
from datasets import TestCustomDataset, MocapDataset
from models.priors import VariationalAutoEncoder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import ConcatDataset
from models.priors import VQVAE
wandb_logger = WandbLogger(save_dir="../checkpoints",
                           project="Mirror_Project", )

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="val_recon_loss",
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
print(custom_data[0][0].shape)
# exit()

layers_order = [38, 70, 100, 128]
latent_dim = 128




def run_vq_prior():
    model = VQVAE(
        encoder_layers=layers_order,
        decoder_layers=list(reversed(layers_order)),
        num_embeddings=50,
        embedding_dim=latent_dim,
        commitment_cost=1.,
        learning_rate=1e-4,
        dataset=custom_data,
        denoise=True,
        use_ema=True,
        mask=False
    )
    # print(model)
    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         max_epochs=10,
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
    num_embeddings=50,
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
for i in range(50):  #(len(codebook)-490):
    # print(len(codebook))
    vector = codebook[i]
    out = decoder(vector.unsqueeze(0)).detach().squeeze(0).view(-1, 2)
    # print(out)
    plot_keypoints(out, title=f"Vector {i}")
    plt.savefig(f"output_images/vq3/codebook/{i}_ema.png")
    # print(out.shape)
exit()
