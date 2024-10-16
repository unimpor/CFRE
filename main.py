import argparse

from cfre import CFRE
from src.models import LLMs, FineGrainedRetriever


def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, help='dataset used, option: ')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')

    # Build retrieval dataset. Note: First consider only training IB.
    # Input: coarsely retrieved graph Output: ground truth Answer

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_retriever)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_retriever)

    # Build Model. Load ibtn, llms, cfre.
    # Options for ibtn: 1) MLP; 2) GNN+MLP
    ibtn = FineGrainedRetriever()
    llms = LLMs()
    cfre = CFRE()

    # Set Optimizer. Optimizer TBD


if __name__ == '__main__':
    main()
