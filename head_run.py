import torch
import torch.nn as nn
from models.networks import LiftNet
from datasets import InternalCustomData  # Abi and Frank
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils import (
    extract_pickle,
    obtain_K,
    obtain_A_from_normal,
    SaveBestModel
)

K = obtain_K()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4

print(DEVICE)
paths = {
    "frank": {
        "data_path": "data/Frank/detections/alphapose-results.json",
        "pickle_path": "data/iccv2023/Frank_0_1999_2000iter.pickle"
    },
    "abi": {
        "data_path": "data/Abi/detections/alphapose-results.json",
        "pickle_path": "data/iccv2023/Abi_0_1999_2000iter.pickle"
    }
}

frank_data = InternalCustomData(paths['frank']['data_path'], vis_list_path="data/Frank/vis_list.json")
abi_data = InternalCustomData(paths['abi']['data_path'], vis_list_path="data/Abi/vis_list.json")

abi_normal = extract_pickle(paths['abi']['pickle_path'])[0]['n_m'][0].detach()
frank_normal = extract_pickle(paths['frank']['pickle_path'])[0]['n_m'][0].detach()

BATCH_SIZE = 200

abi_loader = DataLoader(dataset=abi_data, batch_size=BATCH_SIZE, shuffle=True)
frank_loader = DataLoader(dataset=frank_data, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.MSELoss()
net = LiftNet(layers_list=[52, 256, 128, 128, 78])


# optimizer = torch.optim.Adam()


def train(dataloader, model=net, criterion=loss_fn, batch_size=BATCH_SIZE, epochs=40, normal=abi_normal):
    run = wandb.init()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = list()

    save_model = SaveBestModel()

    for epoch in tqdm(range(epochs)):

        epoch_loss = 0
        epoch_real_loss = 0
        epoch_virtual_loss = 0

        for h, r in dataloader:
            human, reflection = h.to(DEVICE), r.to(DEVICE)

            optimizer.zero_grad()
            # print("yo")
            # print(f"h shape: {human.shape}, r shape: {reflection.shape}")
            output = model(human.view(BATCH_SIZE, -1))

            # print("yo")
            # print(output.shape)

            reshaped_out = output.view(BATCH_SIZE, -1, 3)
            # print(f"reshaped_out shape {reshaped_out.shape}")
            # print(reshaped_out.dtype, K.dtype)
            real_2d = torch.matmul(reshaped_out, K.to(DEVICE))

            mirror_matrix = obtain_A_from_normal(normal).to(DEVICE)
            real_to_virtual = torch.matmul(reshaped_out, mirror_matrix)
            virtual_2d = torch.matmul(real_to_virtual, K.to(DEVICE))

            # losses
            real_loss = criterion(input=real_2d, target=human)
            virtual_loss = criterion(input=virtual_2d, target=reflection)

            total_loss = real_loss + virtual_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_real_loss += real_loss.item()
            epoch_virtual_loss += virtual_loss.item()

        wandb.log(
            {
                "real_loss": epoch_real_loss,
                "virtual_loss": epoch_virtual_loss,
                "total_loss": epoch_loss
            }
        )
        save_model(epoch_loss, epoch, model, optimizer, criterion, path="checkpoints/abi.ckpt")


if __name__ == "__main__":
    print(net)
    # print(next(iter(abi_loader))[0].shape)
    # exit()
    train(
        dataloader=abi_loader,
        model=net,
        criterion=nn.MSELoss(),
        batch_size=BATCH_SIZE,
        epochs=100,
        normal=abi_normal
    )
