# from models.priors import VectorQuantizer
import json
import warnings
import os
import re

warnings.filterwarnings("ignore")
# print(VectorQuantizer)

abi_vis = []
frank_vis = []
pattern = re.compile(r'\^._')

abi_vis = sorted(os.listdir("data/Abi/detections/vis"))
frank_vis = sorted(os.listdir("data/Frank/detections/vis"))[1:]

print(abi_vis); print(frank_vis)
print(abi_vis == frank_vis)
exit()

abi_path = "data/Abi/vis_list.json"
frank_path = "data/Frank/vis_list.json"

with open(abi_path, "w") as file:
    json.dump(abi_vis, file)

with open(frank_path, "w") as file:
    json.dump(frank_vis, file)
# abi_vis = {}

print(abi_vis); print(frank_vis)
print(len(abi_vis)); print(len(frank_vis))
