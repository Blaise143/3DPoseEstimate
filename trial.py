from utils import plot_keypoints
import torch
import json
import matplotlib.pyplot as plt

path = "data/HuMiD-yukagawa-clips/annots/1WJtsn8MvAY+000090+002287/000000.json"

f = open(path)
data = json.load(f)
first = data["annots"][0]["keypoints"]
sec = data["annots"][1]["keypoints"]

kp1 = torch.tensor(first)
kp2 = torch.tensor(sec)
print(kp1)
print(kp2)
kp1 = kp1[:, :2]
kp2 = kp2[:, :2]
print(kp1.shape)
print(kp1)
print(kp2)
# exit()
plot_keypoints(kp1, title="kp1")
plot_keypoints(kp2, title="kp2")
plt.show()
