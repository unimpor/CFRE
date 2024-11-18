import torch
from torch_geometric.data.data import Data
import pickle 

data_comp2 = pickle.load(open(f"train.pkl", "rb"))
data_org = torch.load("train_300.pth")
for dat in data_org:
    result = next((d for d in data_comp2 if d["id"] == data_org["id"]), None)
    


# data1 = torch.load("/home/comp/cscxliu/derek/MultMolOpt/test_300_part0.pth")
# data2 = torch.load("/home/comp/cscxliu/derek/MultMolOpt/test_300_part1.pth")

# data = data1 + data2
# print(len(data))
# post_data_list = []
# for dat in data:
#     dat["graph"] = Data(x=dat["entity_embd"], edge_attr=dat["edge_attr"], edge_index=dat["edge_index"])
#     del dat['entity_embd']
#     del dat['edge_attr']
#     del dat['edge_index']
#     post_data_list.append(dat)
# print(len(post_data_list))
# torch.save(post_data_list, f"test_300.pth")