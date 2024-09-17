from datasets import InternalCustomData, zju_Dataset
from utils import plot_keypoints, plot_J1_overlay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import extract_pickle

abi_pickle = "data/iccv2023/Abi_0_1999_2000iter.pickle"

abi = extract_pickle(abi_pickle)[0]["kp3d"]
# print(len(extract_pickle(abi_pickle)))
print(abi)