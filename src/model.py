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

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        # no output activation as mu mlps are unbounded and alpha mlps positive
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
        # nn.init.xavier_uniform_(self.input_projection.weight)
        # nn.init.zeros_(self.input_projection.bias)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim, self.b + 1))
                for _ in range(self.n_layers)
            ]
        )

        # Weight init
        # for layer_weights in self.weights:
        #     # Xavier/Glorot initialization for each edge type
        #     for i in range(self.b + 1):
        #         nn.init.xavier_uniform_(layer_weights[:, :, i])

    def forward(self, X, A):
        # Input:
        #   - X, dim: batch_size x max_nodes x d
        #   - A, dim: batch_size x max_nodes x max_nodes x b+1
        H = self.input_projection(X)  # dim: batch_size x max_nodes x k
        N = X.shape[1]
        mask = torch.tril(torch.ones(N, N, device=X.device), diagonal=-1)

        for layer in range(self.n_layers):
            W = self.weights[layer]
            tensor_list = []
            for i in range(self.b + 1):  # FIXME could this be tensorized
                E_i = A[:, :, :, i] * mask.unsqueeze(0)
                E_i = E_i + torch.eye(N, device=X.device).unsqueeze(0)
                D_i = torch.sum(E_i, dim=2)
                D_inv_sqrt = torch.diag_embed(torch.pow(D_i + 1e-8, -0.5))
                z = F.relu(D_inv_sqrt @ E_i @ D_inv_sqrt @ H @ W[:, :, i])
                tensor_list.append(z)
            Z = torch.stack(tensor_list, dim=3)  # dim: batch x n x k x b
            H = self.agg(Z, dim=3)  # dim: batch x n x k

        return H


class GraphAF(nn.Module):
    def __init__(self, d=9, b=3, embedding_dim=128):
        super(GraphAF, self).__init__()

        # Parameters
        self.d = d  # number of node types
        self.b = b  # number of edge types
        self.embedding_dim = embedding_dim

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
        N = X.shape[1]

        H = self.rgcn(X, A)  # dim: batch x n x k
        h = self._get_graph_embedding(H, batch_size)  # dim: batch x n x k

        # --- Node part ---
        z_X = X + torch.rand_like(X)
        mu_X = self.mu_node(h)  # dim: batch x n x d
        alpha_X = F.softplus(self.alpha_node(h)) + 1e-8  # dim: batch x n x d
        epsilon_X = (z_X - mu_X) / alpha_X  # dim: batch x n x d

        loss_X = 0.5 * torch.sum(epsilon_X**2, dim=-1) + torch.sum(
            torch.log(alpha_X + 1e-8), dim=-1
        )

        # --- Edge part ---
        z_A = A + torch.rand_like(A)

        # Expand h for all (i,j) pairs
        h_i = h.unsqueeze(2).expand(batch_size, N, N, self.embedding_dim)  # [batch, n, n, k]
        # h_i[:, i, j, :] = h[:, i, :] (graph embedding when generating node i)
        # Node embeddings for pairs
        H_ii = H.unsqueeze(2).expand(batch_size, N, N, self.embedding_dim)  # Node i
        H_ij = H.unsqueeze(1).expand(batch_size, N, N, self.embedding_dim)  # Node j
        # Concatenate: (h_i, H_ii, H_ij)
        edge_features = torch.cat([h_i, H_ii, H_ij], dim=-1)  # [batch, n, n, 3*k]

        mu_A = self.mu_edge(edge_features)  # dim: batch x n x n x b+1
        alpha_A = F.softplus(self.alpha_edge(edge_features)) + 1e-8  # dim: batch x n x n x b+1
        epsilon_A = (z_A - mu_A) / alpha_A  # FIXME - div by zero possibility

        loss_A = 0.5 * torch.sum(epsilon_A**2, dim=-1) + torch.sum(
            torch.log(alpha_A + 1e-8), dim=-1
        )

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

    def generate(self, max_resample=10):
        """Generate a molecule with valency checking."""
        VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1}
        IDX_TO_ATOM = {0: "C", 1: "N", 2: "O", 3: "F", 4: "P", 5: "S", 6: "Cl", 7: "Br", 8: "I"}
        BOND_ORDER = {0: 1, 1: 2, 2: 3, 3: 0}  # single, double, triple, no bond

        device = next(self.parameters()).device

        with torch.no_grad():
            N = 48
            X = torch.zeros(N, self.d, device=device)
            A = torch.zeros(N, N, self.b + 1, device=device)
            A[:, :, self.b] = 1.0  # Initialize all as "no edge"

            valencies_used = [0] * N

            for i in range(N):
                # Add batch dimension
                X_batch = X.unsqueeze(0)
                A_batch = A.unsqueeze(0)

                if i != 0:
                    H = self.rgcn(X_batch, A_batch)  # (1, N, k)
                    H = H.squeeze(0)  # (N, k)
                    h_i = torch.sum(H[:i], dim=0)  # (k,)
                    H_ii = H[i]  # (k,)
                else:
                    h_i = torch.zeros(self.embedding_dim, device=device)
                    H = None

                # Generate node
                epsilon_i = self.epsilon_node.sample().to(device)
                alpha_X = F.softplus(self.alpha_node(h_i)) + 1e-8
                z_i = epsilon_i * alpha_X + self.mu_node(h_i)
                atom_type = torch.argmax(z_i).item()
                X[i] = F.one_hot(
                    torch.tensor(atom_type, device=device), num_classes=self.d
                ).float()

                max_valency_i = VALENCY[IDX_TO_ATOM[atom_type]]

                # Generate edges to previous nodes
                has_edge = False
                for j in range(i):
                    if H is None:
                        continue

                    edge_mlp_input = torch.cat((h_i, H_ii, H[j]), dim=-1)  # (3k,)
                    alpha_A = F.softplus(self.alpha_edge(edge_mlp_input)) + 1e-8
                    mu_A = self.mu_edge(edge_mlp_input)

                    atom_j_type = torch.argmax(X[j]).item()
                    max_valency_j = VALENCY[IDX_TO_ATOM[atom_j_type]]

                    # Sample and resample if valency violated
                    for _ in range(max_resample):
                        epsilon_ij = self.epsilon_edge.sample().to(device)
                        z_ij = epsilon_ij * alpha_A + mu_A
                        bond_idx = torch.argmax(z_ij).item()
                        bond_order = BOND_ORDER[bond_idx]

                        if (
                            valencies_used[i] + bond_order <= max_valency_i
                            and valencies_used[j] + bond_order <= max_valency_j
                        ):
                            break
                    else:
                        # Failed to find valid bond, use no bond
                        bond_idx = self.b
                        bond_order = 0

                    A[i, j] = F.one_hot(
                        torch.tensor(bond_idx, device=device), num_classes=self.b + 1
                    ).float()
                    A[j, i] = A[i, j]
                    valencies_used[i] += bond_order
                    valencies_used[j] += bond_order

                    if bond_order > 0:
                        has_edge = True

                # Stop if no edge to previous subgraph (except first node)
                if i > 0 and not has_edge:
                    X[i] = 0
                    A[i, :] = 0
                    A[:, i] = 0
                    A[i, :, self.b] = 1.0
                    A[:, i, self.b] = 1.0
                    break

            return X, A
