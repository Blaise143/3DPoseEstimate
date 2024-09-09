from datasets import InternalCustomData
from utils import plot_keypoints, plot_J1_overlay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib
os.environ.pop('SESSION_MANAGER', None)
frank_path = "data/Frank/detections/alphapose-results.json"
abi_path = "data/Abi/detections/alphapose-results.json"

img_path = "data/Abi/frames/00000000.jpg"

im = plt.imread(img_path)

data = InternalCustomData(abi_path, vis_list_path="data/Abi/vis_list.json")
frank = InternalCustomData(frank_path, vis_list_path="data/Frank/vis_list.json")

# print(data[0][0
human = data[10][0]
reflection = data[10][1]

f_0 = frank[1990][0]
f_1 = frank[1990][1]
print(f"shape: {f_1.shape}")

# print(human)
# exit()
# plot_J1_overlay(data=human, mirror_data=reflection, image_path="data/Abi/frames/00000010.jpg")

# plt.savefig("output_images/sec_viz.png")
# plot = plt.imshow(im)
plot_keypoints(f_0, kind="J1", title="h")
plt.savefig("output_images/visualizations/h.png")
plot_keypoints(f_1, kind="J1", title="r")
plt.savefig("output_images/visualizations/r.png")
# plt.show()
