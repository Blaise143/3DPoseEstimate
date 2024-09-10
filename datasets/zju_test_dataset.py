import torch
from torch.utils.data import Dataset
import os
import json


class TestCustomDataset(Dataset):
    """
    This is a custom dataset for zju-m-seq1
    """

    def __init__(self, path: str, transform=None) -> None:
        self.transform = transform
        json_files = [path+"/"+file for file in sorted(os.listdir(
            path)) if file.endswith(".json")]
        real = []
        virtual = []
        for file in json_files:
            data = json.load(open(file))
            data_0 = data["annots"][0]["keypoints"]
            data_1 = data["annots"][1]["keypoints"]

            real.append(data_0)
            virtual.append(data_1)

        real, virtual = torch.tensor(
            real)[:, :-6, :], torch.tensor(virtual)[:, :-6, :]

        # scaling real
        real = self.scale_by_height(real)
        # scaling virtual
        virtual = self.scale_by_height(virtual)

        real_center = real[:, 8, :].unsqueeze(1)
        virtual_center = virtual[:, 8, :].unsqueeze(1)

        real, virtual = real - real_center, virtual - virtual_center

        real, virtual = real[:, :, :2], virtual[:, :, :2]
        assert real.shape == virtual.shape

        # shuffling
        indices = torch.randperm(real.shape[0])
        real = real[indices]
        virtual = virtual[indices]

        self.real, self.virtual = real.reshape(
            real.shape[0], -1), virtual.reshape(virtual.shape[0], -1)

    @staticmethod
    def scale_by_height(data: torch.Tensor) -> torch.Tensor:
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

        if self.transform:
            real = self.transform(real)
            virtual = self.transform(virtual)

        return real, virtual

    def __len__(self):
        return self.real.shape[0]


if __name__ == "__main__":
    pth = "../data/zju-m-seq1/annots/3"

    pth3 = "../data/zju-m-seq1/annots/5"
    pth4 = "../data/zju-m-seq1/annots/6"

    from torch.utils.data import ConcatDataset
    dataset = TestCustomDataset(path=pth)

    # dataset2 = TestCustomDataset(path=pth2)
    # datase3 = TestCustomDataset(path=pth3)
    dataset4 = TestCustomDataset(path=pth4)
    # print(len(dataset))
    # print(dataset[0][0])
    combined_dataset = ConcatDataset([dataset, dataset4])
    print(len(combined_dataset))
