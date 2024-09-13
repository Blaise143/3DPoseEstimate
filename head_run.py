import torch
import torch.nn as nn
from models.networks import LiftNet
from datasets import InternalCustomData # Abi and Frank

frank_path = "data/Frank/detections/alphapose-results.json"
abi_path = "data/Abi/detections/alphapose-results.json"

frank_data = InternalCustomData(frank_path, vis_list_path="data/Frank/vis_list.json")
abi_data = InternalCustomData(abi_path,vis_list_path="data/Abi/vis_list.json")




print(frank_data[0])
exit()
layers_list = (52, 256, 256, 156, 78)
net = LiftNet()
rand = torch.rand(12,52)
print(rand)
print(net(rand))