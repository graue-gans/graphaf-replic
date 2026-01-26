import torch
from torch.utils.data import DataLoader

from dataset import ZINCDataset, collate_graphs
from model import GraphAF


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    dataset = ZINCDataset("data/250k_rndm_zinc_drugs_clean_3.csv")
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_graphs,  # Custom batching logic
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
    )

    # Training loop
    for epoch in range(num_epochs):  # while theta is not converged
        for batch in data_loader:
            X = batch["X"]  # dim: batch_size x max_nodes x d
            A = batch["A"]  # dim: batch_size x max_nodes x max_nodes x b+1

            optimizer.zero_grad()
            loss = model(X, A)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
