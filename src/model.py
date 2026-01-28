import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# pyright: reportPossiblyUnboundVariable=false


class MLP(nn.Module):
    """Base class for the node and edge MLPs using tanh nonlinearities."""

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


class RGCN(nn.Module):
    """Relational Graph Convolutional Network (R-GCN) with autoregressive masking of the adjacency tensor."""

    def __init__(self, d, embedding_dim, b, n_layers=3, agg=torch.sum):
        super(RGCN, self).__init__()

        self.embedding_dim = embedding_dim
        self.b = b
        self.n_layers = n_layers
        self.agg = agg

        # Initial embedding: d to k
        self.input_projection = nn.Linear(d, embedding_dim)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim, self.b + 1))
                for i in range(self.n_layers)
            ]
        )

    def forward(self, X, A):
        # Input:
        #   - X, dim: batch_size x max_nodes x d
        #   - A, dim: batch_size x max_nodes x max_nodes x b+1
        H = self.input_projection(X)  # dim: batch_size x max_nodes x k
        N = X.shape[1]
        mask = torch.tril(torch.ones(N, N), diagonal=-1)

        for layer in range(self.n_layers):
            W = self.weights[layer]
            tensor_list = []
            for i in range(self.b + 1):  # FIXME could this be tensorized
                E_i = A[:, :, :, i] * mask.unsqueeze(0)
                E_i = E_i + torch.eye(N).unsqueeze(0)
                D_i = torch.sum(E_i, dim=2)
                D_inv_sqrt = torch.diag_embed(torch.pow(D_i, -0.5))
                z = F.relu(D_inv_sqrt @ E_i @ D_inv_sqrt @ H @ W[:, :, i])
                tensor_list.append(z)
            Z = torch.stack(tensor_list, dim=3)  # dim: batch x n x k x b
            H = self.agg(Z, dim=3)  # dim: batch x n x k

        return H


class GraphAF(nn.Module):
    def __init__(self, d=9, b=3, embedding_dim=128, max_nodes=48):
        super(GraphAF, self).__init__()

        # Parameters
        self.d = d  # number of node types
        self.b = b  # number of edge types
        self.embedding_dim = embedding_dim
        self.N = max_nodes

        # Distributions; FIXME - naming
        self.epsilon_node = MultivariateNormal(torch.zeros(d), torch.eye(d))
        self.epsilon_edge = MultivariateNormal(torch.zeros(b + 1), torch.eye(b + 1))

        # Node and Edge MLPs
        self.mu_node = MLP(embedding_dim, 2 * embedding_dim, d)
        self.alpha_node = MLP(embedding_dim, 2 * embedding_dim, d)
        self.mu_edge = MLP(3 * embedding_dim, 2 * 3 * embedding_dim, b + 1)
        self.alpha_edge = MLP(3 * embedding_dim, 2 * 3 * embedding_dim, b + 1)

        # Autoregressive R-GCN
        self.rgcn = RGCN(d, embedding_dim, b)

    def forward(self, X, A):
        # Input:
        #   - X, dim: batch_size x max_nodes x d
        #   - A, dim: batch_size x max_nodes x max_nodes x b+1
        batch_size = X.shape[0]

        H = self.rgcn(X, A)  # dim: batch x n x k
        h = self._get_graph_embedding(H, batch_size)  # dim: batch x n x k

        # --- Node part ---
        z_X = X + torch.rand_like(X)
        mu_X = self.mu_node(h)  # dim: batch x n x d
        alpha_X = F.softplus(self.alpha_node(h)) + 1e-8  # dim: batch x n x d
        epsilon_X = (z_X - mu_X) / alpha_X  # dim: batch x n x d

        loss_X = -torch.sum(self.epsilon_node.log_prob(epsilon_X), dim=-1) - torch.log(
            torch.prod(alpha_X, dim=-1)
        )  # dim: batch x n

        # --- Edge part ---
        z_A = A + torch.rand_like(A)

        # Expand h for all (i,j) pairs
        h_i = h.unsqueeze(2).expand(
            batch_size, self.N, self.N, self.embedding_dim
        )  # [batch, n, n, k]
        # h_i[:, i, j, :] = h[:, i, :] (graph embedding when generating node i)
        # Node embeddings for pairs
        H_ii = H.unsqueeze(2).expand(batch_size, self.N, self.N, self.embedding_dim)  # Node i
        H_ij = H.unsqueeze(1).expand(batch_size, self.N, self.N, self.embedding_dim)  # Node j
        # Concatenate: (h_i, H_ii, H_ij)
        edge_features = torch.cat([h_i, H_ii, H_ij], dim=-1)  # [batch, n, n, 3*k]

        mu_A = self.mu_edge(edge_features)  # dim: batch x n x n x b+1
        alpha_A = F.softplus(self.alpha_edge(edge_features)) + 1e-8  # dim: batch x n x n x b+1
        epsilon_A = (z_A - mu_A) / alpha_A  # FIXME - div by zero possibility

        loss_A = -torch.sum(self.epsilon_edge.log_prob(epsilon_A), dim=-1) - torch.log(
            torch.prod(alpha_A, dim=-1)
        )  # dim: batch x n x n

        loss = (torch.sum(loss_X) + torch.sum(loss_A)) / batch_size
        return loss

    def _get_graph_embedding(self, H, batch_size):
        """
        Compute graph embedding h_i for each node i.
        h_i = sum(H[0:i]) represents the subgraph G_i containing nodes 0 to i-1.
        For node 0, h_0 = 0 (empty graph).
        """
        H_cumsum = torch.cumsum(H, dim=1)  # dim: batch x n x k
        # Shift right: h[i] = sum(H[0:i])
        h = torch.cat(
            [
                torch.zeros(batch_size, 1, self.embedding_dim, device=H.device),
                H_cumsum[:, :-1, :],
            ],
            dim=1,
        )  # dim: batch x n x k
        return h

    def generate(self):
        """TODO - add (1 x ...) batch dim for rgcn"""
        with torch.no_grad():
            # Empty init; FIXME - check if this is the way to do it
            X = torch.zeros(self.N, self.d)
            A = torch.zeros(self.N, self.N, self.b + 1)

            for i in range(self.N):
                if i != 0:
                    H_i = self.rgcn(X, A)  # dim: n x k
                    H_ii = H_i[i, :]  # dim: k
                    h_i = torch.sum(H_i, dim=0)  # dim: k
                else:
                    h_i = torch.zeros(self.embedding_dim)

                epsilon_i = self.epsilon_node.sample()
                alpha_X = F.softplus(self.alpha_node(h_i)) + 1e-8
                z_i = epsilon_i * alpha_X + self.mu_node(h_i)
                X[i, :] = F.one_hot(torch.argmax(z_i), num_classes=self.d)  # dim: d

                for j in range(i):  # corrected from (i - 1)
                    epsilon_ij = self.epsilon_edge.sample()
                    edge_mlp_input = torch.cat((h_i, H_ii, H_i[j, :]), dim=-1)  # dim: 3k
                    alpha_A = F.softplus(self.alpha_edge(edge_mlp_input)) + 1e-8
                    z_ij = epsilon_ij * alpha_A + self.mu_edge(edge_mlp_input)
                    A[i, j, :] = F.one_hot(torch.argmax(z_ij), num_classes=self.b + 1)
