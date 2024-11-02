import torch
from torch_geometric.data.data import Data

data = torch.load("val_300_org.pth")
post_data_list = []
for dat in data:
    dat["graph"] = Data(x=dat["entity_embd"], edge_attr=dat["edge_attr"], edge_index=dat["edge_index"])
    del dat['entity_embd']
    del dat['edge_attr']
    del dat['edge_index']
    post_data_list.append(dat)
torch.save(post_data_list, f"val_300.pth")