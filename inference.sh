#!/bin/bash

for i in warmup lora_w.o.gumbel 
do
  python main.py --proj_name $i
done

# for i in frozen full-finetune lora lora_w.o.gumbel warmup
# do
#   python main.py --proj_name $i --gnn graphsage
# done