import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset


class ZINCDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        self.atom_types = {"C": 0, "N": 1, "O": 2, "F": 3, "P": 4, "S": 5, "Cl": 6, "Br": 7, "I": 8}
        self.d = len(self.atom_types)
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
        }
        self.b = len(self.bond_types)

        self.graphs = self._load_graphs()

    def _load_graphs(self):
        """Load and convert all molecules from smiles to graph format"""
        graphs = []
        for smiles in self.df["smiles"]:
            # Convert smiles to molecule
            mol = Chem.MolFromSmiles(smiles)

            # Kekulize molecule
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception as e:
                print(f"Failed to kekulize {smiles}: {e}")
                continue  # skip this molecule

            # Convert molecule to graph
            if mol is not None:
                graph = self._mol_to_graph(mol)
                graphs.append(graph)

        return graphs

    def _mol_to_graph(self, mol):
        """
        Convert RDKit molecule to graph representation
        Returns: dict with adjacency tensor A and node features X
        """
        N = mol.GetNumAtoms()

        # --- Feature matrix ---
        # FIXME - N should be padded for batching but probably here but in the batching logic
        X = torch.zeros((N, self.d), dtype=torch.int8)
        # TODO - order with bfs
        for i, atom in enumerate(mol.GetAtoms()):
            idx = self.atom_types[atom.GetSymbol()]  # get index for element
            X[i, idx] = 1

        # ---- Adjacency tensor ----
        A = torch.zeros((N, N, self.b + 1), dtype=torch.int8)
        # ASSUME - bfs ordering
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            idx = self.bond_types[bond.GetBondType()]
            A[i, j, idx] = 1
            A[j, i, idx] = 1

        return {"X": X, "A": A}

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(batch):
    max_nodes = 48
    X_list = []
    A_list = []

    for graph in batch:
        n = graph["X"].shape[0]

        X_padded = torch.zeros(max_nodes, graph["X"].shape[1])
        X_padded[:n] = graph["X"]
        X_list.append(X_padded)

        A_padded = torch.zeros(max_nodes, max_nodes, graph["A"].shape[2])
        A_padded[:n, :n, :] = graph["A"]
        A_list.append(A_padded)

    X_batch = torch.stack(X_list, dim=0)  # dim: batch_size x max_nodes x d
    A_batch = torch.stack(A_list, dim=0)  # dim: batch_size x max_nodes x max_nodes x b+1
    return {"X": X_batch, "A": A_batch}
