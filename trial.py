from datasets import InternalCustomData, zju_Dataset
from utils import plot_keypoints, plot_J1_overlay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import extract_pickle

# abi_pickle = "data/iccv2023/Abi_0_1999_2000iter.pickle"
#
# abi = extract_pickle(abi_pickle)[0]["kp3d"]
# print(len(extract_pickle(abi_pickle)))
# print(abi)
import torch

src = torch.arange(1,11).reshape((2,5))
print(src)
index = torch.tensor([[0,1,2,0]])

zeros = torch.zeros(3,5, dtype=src.dtype)
print(zeros)
zeros_scatter = zeros.scatter_(0, index, src)
print(zeros_scatter)