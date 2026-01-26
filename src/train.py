import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from dataset import ZINCDataset, collate_graphs
from model import GraphAF


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    epsilon_node = MultivariateNormal(torch.zeros(d), torch.eye(d))
    epsilon_edge = MultivariateNormal(torch.zeros(b + 1), torch.eye(b + 1))
    P = 12

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

            # sample molecule from dataset and get graph size N
            # convert molecule to graph with BFS ordering

            # FOR LOOP DRAFT
            node_loss = 0
            edge_loss = 0
            for i in range(N):
                # Message passing
                H_i = model.rgcn().forward(subgraph)  # dim: n x k
                h_i = torch.sum(H_i, dim=0)  # dim: k

                #
                z_i = X[i, :] + torch.rand(d)  # dim: d

                mu_i = model.mu_node.forward(h_i)  # dim: d
                alpha_i = model.alpha_node.forward(h_i)  # dim: d

                epsilon_i = (z_i - mu_i) * (1 / alpha_i)  # dim: d

                # - sum log p(epsilon_i) = - log prod p(epsilon_i)
                node_loss += -torch.sum(epsilon_node.log_prob(epsilon_i)) - torch.log(torch.prod(1 / alpha_i))

                for j in range(max(1, i - P), i - 1):
                    z_ij = A[i, j, :] + torch.rand(b + 1)  # dim: b+1

                    mu_ij = model.mu_edge.forward(h_i, H_i[i], H_i[j])  # dim: b+1
                    alpha_ij = model.alpha_edge.forward(h_i, H_i[i], H_i[j])  # dim: b+1

                    epsilon_ij = (z_ij - mu_ij) * (1 / alpha_ij)  # dim: b+1

                    # - sum log p(epsilon_ij) = - log prod p(epsilon_ij)
                    edge_loss += -torch.sum(epsilon_edge.log_prob(epsilon_ij)) - torch.log(torch.prod(1 / alpha_ij))

            loss += node_loss + edge_loss


if __name__ == "__main__":
    main()
