import torch
from torch_geometric.data.data import Data

data1 = torch.load("/home/comp/cscxliu/derek/MultMolOpt/train_300_part0.pth")
data2 = torch.load("/home/comp/cscxliu/derek/MultMolOpt/train_300_part1.pth")
data3 = torch.load("/home/comp/cscxliu/derek/MultMolOpt/train_300_part2.pth")
data = data1 + data2 + data3
print(len(data))
post_data_list = []
for dat in data:
    if len(dat['relevant_idx']) == 0:
        continue
    dat["graph"] = Data(x=dat["entity_embd"], edge_attr=dat["edge_attr"], edge_index=dat["edge_index"])
    del dat['entity_embd']
    del dat['edge_attr']
    del dat['edge_index']
    post_data_list.append(dat)
print(len(post_data_list))
torch.save(post_data_list, f"train_300.pth")