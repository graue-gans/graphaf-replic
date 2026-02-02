import torch
from rdkit import Chem, RDLogger

from model import GraphAF

RDLogger.DisableLog("rdApp.*")

MODEL_PATH = "checkpoints/graphaf_finetune_best.pt"  # FILL IN
DATASET_PATH = "data/Tg_SMILES_class_pid_polyinfo_median.csv"  # FILL IN (for novelty check)
NUM_SAMPLES = 10000
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

IDX_TO_ATOM = {0: "C", 1: "N", 2: "O", 3: "F", 4: "P", 5: "S", 6: "Cl", 7: "Br", 8: "I"}
IDX_TO_BOND = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

# SMARTS patterns for polymerizable functional groups
POLYMERIZABLE_GROUPS = {
    "vinyl": "[CH2]=[CH]",  # Vinyl group
    "acrylate": "[CH2]=[CH]C(=O)O",  # Acrylate
    "methacrylate": "[CH2]=[C](C)C(=O)O",  # Methacrylate
    "epoxy": "C1OC1",  # Epoxide ring
    "styrene": "[CH2]=[CH]c1ccccc1",  # Styrene-type
    "diene": "[CH2]=[CH][CH]=[CH2]",  # 1,3-diene (butadiene-type)
    "isocyanate": "N=C=O",  # Isocyanate
    "amine": "[NH2]",  # Primary amine (for polyamides)
    "carboxylic_acid": "C(=O)[OH]",  # Carboxylic acid (for polyesters/polyamides)
    "alcohol": "[CH2][OH]",  # Primary alcohol (for polyesters)
    "thiol": "[SH]",  # Thiol (for thiol-ene)
}


def check_polymerizable(mol):
    """Check if molecule contains polymerizable functional groups."""
    found_groups = []
    for name, smarts in POLYMERIZABLE_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            found_groups.append(name)
    return found_groups


def graph_to_mol(X, A):
    """Convert graph (X, A) to RDKit molecule."""
    X = X.argmax(dim=-1)
    A = A.argmax(dim=-1)

    # Find actual number of atoms (non-padding nodes that are connected)
    n_atoms = 0
    for i in range(X.shape[0]):
        # Check if node has valid atom type
        if X[i].item() >= len(IDX_TO_ATOM):
            break
        # First node or has edge to previous nodes
        if i == 0:
            n_atoms = 1
        elif (A[i, :i] < len(IDX_TO_BOND)).any().item():
            n_atoms = i + 1
        else:
            break

    if n_atoms == 0:
        return None

    mol = Chem.RWMol()

    # Add atoms
    for i in range(n_atoms):
        atom_idx = X[i].item()
        if atom_idx not in IDX_TO_ATOM:
            return None
        atom = Chem.Atom(IDX_TO_ATOM[atom_idx])
        mol.AddAtom(atom)

    # Add bonds
    for i in range(n_atoms):
        for j in range(i):
            bond_idx = A[i, j].item()
            if bond_idx < len(IDX_TO_BOND):  # Not "no bond"
                mol.AddBond(j, i, IDX_TO_BOND[bond_idx])

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def load_training_smiles(csv_path):
    """Load SMILES from training set for novelty check."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    smiles_set = set()
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smiles_set.add(Chem.MolToSmiles(mol))
    return smiles_set


def evaluate(model, num_samples, training_smiles):
    """Generate molecules and compute metrics."""
    model.eval()

    valid_mols = []
    valid_smiles = []
    polymerizable_count = 0
    group_counts = {name: 0 for name in POLYMERIZABLE_GROUPS}

    print(f"Generating {num_samples} molecules...")
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{num_samples}")

        with torch.no_grad():
            X, A = model.generate()

        mol = graph_to_mol(X.cpu(), A.cpu())
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))

            # Check for polymerizable groups
            found_groups = check_polymerizable(mol)
            if found_groups:
                polymerizable_count += 1
                for group in found_groups:
                    group_counts[group] += 1

    # Metrics
    validity = len(valid_mols) / num_samples

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    novel_smiles = unique_smiles - training_smiles
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    # Polymerizable ratio
    polymerizable_ratio = polymerizable_count / len(valid_mols) if valid_mols else 0

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "polymerizable_ratio": polymerizable_ratio,
        "num_valid": len(valid_mols),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
        "num_polymerizable": polymerizable_count,
        "group_counts": group_counts,
    }


def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}")
    model = GraphAF(d=9, b=3, embedding_dim=128).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load training SMILES
    print(f"Loading training SMILES from {DATASET_PATH}")
    training_smiles = load_training_smiles(DATASET_PATH)
    print(f"  Loaded {len(training_smiles)} training molecules")

    # Evaluate
    metrics = evaluate(model, NUM_SAMPLES, training_smiles)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Validity:       {metrics['validity']:.4f} ({metrics['num_valid']}/{NUM_SAMPLES})")
    print(
        f"Uniqueness:     {metrics['uniqueness']:.4f} ({metrics['num_unique']}/{metrics['num_valid']})"
    )
    print(
        f"Novelty:        {metrics['novelty']:.4f} ({metrics['num_novel']}/{metrics['num_unique']})"
    )
    print("-" * 50)
    print(
        f"Polymerizable:  {metrics['polymerizable_ratio']:.4f} ({metrics['num_polymerizable']}/{metrics['num_valid']})"
    )
    print("\nFunctional group breakdown:")
    for group, count in metrics["group_counts"].items():
        if count > 0:
            print(f"  {group}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
