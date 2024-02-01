import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import json


class MocapDataset(Dataset):
    def __init__(self, path: str, transform=None) -> None:
        super().__init__()
        # json_files =
        real = []
        virtual = []
        self.transform = transform

        for root, dirs, files in os.walk(path):
            exclude_folder = "_ix0slYCf7E+001140+002600"
            if exclude_folder in dirs:
                dirs.remove(exclude_folder)
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                        # print(f"data {data}")
                        if len(data["annots"]) == 2:
                            data_0 = data["annots"][0]["keypoints"]
                            # print(f"data_0: {(data_0[0])}")
                            data_1 = data["annots"][1]["keypoints"]

                            real.append(data_0)
                            virtual.append(data_1)

        # Sentinel value for missing keypoints
        missing_value = -999

        real, virtual = torch.tensor(
            real)[:, :-6, :], torch.tensor(virtual)[:, :-6, :]

        # Masking before scaling
        real_mask = (real != 0).all(dim=-1, keepdim=True).float()
        virtul_mask = (virtual != 0).all(dim=-1, keepdim=True).float()

        # scaling real and virtual
        real = self.scale_by_height(real)
        virtual = self.scale_by_height(virtual)

        # real[real_mask] = missing_value
        # virtual[virtul_mask] = missing_value

        # check this
        real_center = real[:, 8, :].unsqueeze(1)
        virtual_center = virtual[:, 8, :].unsqueeze(1)

        real, virtual = real - real_center, virtual - virtual_center

        # restoring hip keypoint to (0,0)
        # real[:, 8] = 0.0
        # virtual[:, 8] = 0.0

        real, virtual = real[:, :, :2], virtual[:, :, :2]
        assert real.shape == virtual.shape

        # shuffling
        indices = torch.randperm(real.shape[0])
        real = real[indices]
        virtual = virtual[indices]
        self.real_mask = real_mask[indices]
        self.virtual_mask = virtul_mask[indices]

        self.real, self.virtual = real.reshape(
            real.shape[0], -1), virtual.reshape(virtual.shape[0], -1)

    def scale_by_height(self, data: torch.Tensor) -> torch.Tensor:
        # expects data.shape = (batch_size, N, 2)
        min_vals = torch.min(data, dim=1, keepdim=True).values
        max_vals = torch.max(data, dim=1, keepdim=True).values
        size = max_vals - min_vals
        max_size = torch.max(size, dim=2, keepdim=True).values
        scale_factor = 1.0 / torch.clamp(max_size, min=1e-6)
        data = (data - min_vals) * scale_factor
        return data

    def __getitem__(self, idx):
        real = self.real[idx]
        virtual = self.virtual[idx]
        real_mask = self.real_mask[idx]
        virtual_mask = self.virtual_mask[idx]

        if self.transform:
            real = self.transform(real)
            virtual = self.transform(virtual)

        return (real, real_mask), (virtual, virtual_mask)

    def __len__(self):
        return self.real.shape[0]


if __name__ == "__main__":
    path = "data/HuMiD-yukagawa-clips"
    data = MocapDataset(path)
    print(len(data))
    print(data[0])
    print(data[0][0].shape)
