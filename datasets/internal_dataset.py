import torch
from torch import Tensor
from torch.utils.data import Dataset
import json
from collections import OrderedDict
from typing import Tuple, List, Union

keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck',
                  'hip', 'left_toe', 'right_toe', 'left_small_toe', "right_small_toe", "left_heel", "right_heel"]


class CustomDataset(Dataset):
    """
    Custom dataset for internal video data, meaning Abi and Frank frames
    """

    def __init__(self, path: str, vis_list_path: str = "../data/Abi/vis_list.json"):
        data = json.load(open(path))
        vis_list = json.load(open(vis_list_path))
        df = dict()
        for item in data:
            if item["image_id"] in vis_list:
                if item["image_id"] in df.keys():
                    # print(item["image_id"])
                    df[item["image_id"]].append(item["keypoints"])
                else:
                    df[item["image_id"]] = [item["keypoints"]]

        df = {key: torch.tensor(val).view((-1, 26, 3))[:, :, :2] for key, val in df.items()}
        df = OrderedDict(sorted(df.items()))
        new_df = dict()
        for key, val in df.items():
            # order the vals
            sorted_tensor = self.sort_by_distance(val)
            new_df[key] = sorted_tensor
        humans, reflections = [], []
        for key, val in new_df.items():
            # print(key)
            humans.append(new_df[key][0][1].tolist())
            reflections.append(new_df[key][1][1].tolist())

        humans = torch.tensor(humans)
        reflections = torch.tensor(reflections)

        # scaling by height
        humans, reflections = self.scale_by_height(humans), self.scale_by_height(reflections)

        # centering at hip which is located at index 19
        humans, reflections = self.center_at_hip(humans), self.center_at_hip(reflections)
        assert humans.shape == reflections.shape, "There should be an equal number of real and reflection poses"
        self.humans = humans
        self.reflections = reflections

    @staticmethod
    def scale_by_height(data: torch.Tensor) -> torch.Tensor:
        # expects data.shape = (batch_size, N, 2)
        min_vals = torch.min(data[..., 1], dim=1, keepdim=True).values
        max_vals = torch.max(data[..., 1], dim=1, keepdim=True).values
        height = max_vals - min_vals
        # max_size = torch.max(size, dim=2, keepdim=True).values

        scale_factor = 1.0 / torch.clamp(height, min=1e-6)
        data[..., 1] = (data[..., 1] - min_vals) * scale_factor
        data[..., 0] = (data[..., 0] - min_vals) * scale_factor
        # data = (data - torch.min(data, dim=1, keepdim=True).values) * scale_factor

        return data

    @staticmethod
    def center_at_hip(data: torch.tensor, hip_idx: int = 19) -> torch.tensor:
        # also expects data.shape = (batch_size, N, 2)
        centers = data[:, hip_idx]
        centered = data - centers.unsqueeze(1)
        return centered

    @staticmethod
    def euclidean_distance(point1: torch.tensor, point2: torch.tensor) -> float:
        x_1, y_1 = point1
        x_2, y_2 = point2
        squared_diff_x = (x_1 - x_2) ** 2
        squared_diff_y = (y_1 - y_2) ** 2
        out = torch.sqrt(squared_diff_x + squared_diff_y).item()
        return out

    def sort_by_distance(self, tensor: torch.tensor, fixed_indices: Union[List[int], None] = None):
        if fixed_indices is None:
            fixed_indices = [18, 19]

        acc = dict()
        neck, hip = fixed_indices

        for i in tensor:
            distance = self.euclidean_distance(i[neck], i[hip])
            acc[distance] = i
        distances_added = [(dist, val) for dist, val in dict(reversed(sorted(acc.items()))).items()]
        top_2 = distances_added[:2]
        return top_2

    def __len__(self) -> int:
        return len(self.humans)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        human = self.humans[idx]
        reflection = self.reflections[idx]
        return human, reflection


if __name__ == "__main__":
    abi_path = "../data/Abi/detections/alphapose-results.json"
    frank_path = "../data/Frank/detections/alphapose-results.json"

    some_data = json.load(open(frank_path))
    # print(some_data[1]["keypoints"])

    custom = CustomDataset(frank_path)

    print(len(custom))
    print(custom[0])
    # some_abi_data = json.load(open(abi_path))
    custom_b = CustomDataset(abi_path)
    # print(len(custom_b))
    print(custom_b[0])
    print("Printed stuff")
