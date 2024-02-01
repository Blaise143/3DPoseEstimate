from utils import plot_keypoints
import torch
import json
import matplotlib.pyplot as plt
from datasets import MocapDataset

tensor = torch.tensor([[1, 0, 3], [0, 2, 0], [4, 5, 6], [0, 0, 0], [7, 8, 9]])
print(tensor)
mask = (tensor != 0).int()
print(mask)
print(f"mask shape:{mask.shape}")
# masked = tensor[mask]
# print(masked)
# # TODO: COMMENTING STARTS HERE
path = "data/HuMiD-yukagawa-clips"
data = MocapDataset(path)

first = data[230][0][0].view(-1, 2)
first_mask = data[230][0][1]
sec = data[220][1][0].view(-1, 2)
sec_mask = data[220][1][1]
print(f"first: {first.view(-1)}")
print(list(zip(sec.view(-1, 2).tolist(), sec_mask.tolist())))
print(list(zip(first.view(-1, 2).tolist(), first_mask.tolist())))
# print(f"sec shape: {sec.shape}, sec_mask shape: {first_mask.shape}")
# print(f"first_mask: {first_mask.view(-1)}")

plot_keypoints(first, title="real")
plot_keypoints(sec, title="virtual")
plt.show()
# COMMENTING ENDS HERE

# path = "data/HuMiD-yukagawa-clips/annots/1WJtsn8MvAY+000090+002287/000000.json"
# path = "data/HuMiD-yukagawa-clips/annots/2P2nIhNTZgU+011262+011722/000004.json"
# path = "data/HuMiD-yukagawa-clips/annots/2P2nIhNTZgU+011262+011722/000458.json"
# path = "data/HuMiD-yukagawa-clips/annots/xwgefd1_csA+003280+004320/000013.json"
# path = "data/HuMiD-yukagawa-clips/annots/xwgefd1_csA+003280+004320/000130.json"
# path = "data/HuMiD-yukagawa-clips/annots/xwgefd1_csA+003280+004320/000103.json"
# # path = "data/HuMiD-yukagawa-clips/annots/X2X5zUfge5c+015920+017490/000004.json"
# f = open(path)
# data = json.load(f)
# first = data["annots"][0]["keypoints"]
# sec = data["annots"][1]["keypoints"]

# kp1 = torch.tensor(first)
# kp2 = torch.tensor(sec)
# # print(kp1)
# # print(kp2)
# kp1 = kp1[:, :2]  # [:-6]
# kp2 = kp2[:, :2]  # [:-6]

# # print(kp1.shape)
# print("____")
# print(kp1)
# print("___")
# print(kp2)
# print(f"kp1 shape: {kp1.shape}")
# # exit()
# # TODO: VERIFY THIS
# plot_keypoints(kp1, title="real", kind="J2")
# plt.show()
# plot_keypoints(kp2, title="virtual", kind="J2")
# plt.show()
