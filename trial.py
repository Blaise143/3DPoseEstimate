from datasets import InternalCustomData, zju_Dataset
from utils import plot_keypoints, plot_J1_overlay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

os.environ.pop('SESSION_MANAGER', None)
path = "data/zju-m-seq1/annots/3"
data = zju_Dataset(path)
# print(len(data))
h_0 = data[0][0].view(-1,2)
r_0 = data[0][1].view(-1, 2)


print(h_0.shape)
plot_keypoints(h_0)
plt.savefig("output_images/visualizations/first.png")
plot_keypoints(r_0)
plt.savefig("output_images/visualizations/sec.png")