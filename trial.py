from datasets import InternalCustomData
from utils import plot_keypoints, plot_J1_overlay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
frank_path = "data/Frank/detections/alphapose-results.json"
abi_path = "data/Abi/detections/alphapose-results.json"

img_path = "data/Abi/frames/00000000.jpg"

im = plt.imread(img_path)

data = InternalCustomData(abi_path, vis_list_path="data/Abi/vis_list.json")
frank = InternalCustomData(frank_path, vis_list_path="data/Frank/vis_list.json")

# print(data[0][0])
human = data[1900][0]
reflection = data[1900][1]

f_0 = frank[1990][0]
f_1 = frank[1990][1]
print(f"shape: {f_1.shape}")

# print(human)
plot_J1_overlay(data=f_0, mirror_data=f_1, image_path="data/Frank/frames/00001990.jpg")
plt.show()
# plot = plt.imshow(im)
# plot_keypoints(human, kind="J1", title="Yo")
# plot_keypoints(reflection, kind="J1", title="Yo")
# plt.show()
