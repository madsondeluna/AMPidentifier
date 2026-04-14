# amp_identifier/feature_extraction.py
#
# Macrel-inspired feature set (Santos-Junior et al. 2020, Bhadra et al. 2018).
#
# Feature groups (22 total):
#   Global descriptors (6): Charge, pI, InstabilityInd, AliphaticInd, BomanInd, HydrophRatio
#   Hydrophobic moment (1): Eisenberg scale, angle=100° (helix)
#   Grouped AAC (9):        fraction of residues in 9 functional groups
#   FET local (3):          relative position of first residue in each FET group
#   Solvent access local (3): relative position of first residue in each SA group

import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Free Energy of Transition groups (Von Heijne & Blomberg 1979)
# Used by Macrel for local positional features.
# D1 = relative position of first residue in each group (fraction of sequence length).
FET_GROUPS = {
    "FET_low":  set("ILVWAMGT"),  # lowest FET: hydrophobic, membrane-preferring
    "FET_mid":  set("FYSQCN"),    # intermediate FET
    "FET_high": set("PHKEDR"),    # highest FET: hydrophilic, membrane-avoiding
}

# Solvent accessibility groups (Bhadra et al. 2018)
# D1 = relative position of first residue in each group.
SA_GROUPS = {
    "SA_buried": set("ALFCGIVW"),
    "SA_exposed": set("RKQEND"),
    "SA_inter":   set("MSPTHY"),
}

# Amino acid functional groups (Jhong et al. 2019; Nagarajan et al. 2019)
# f_ prefix denotes fraction of residues belonging to the group.
AA_GROUPS = {
    "acidic":    set("DE"),
    "basic":     set("KRH"),
    "polar":     set("STNQ"),
    "nonpolar":  set("AVLIMFYWP"),
    "aliphatic": set("AVLIM"),
    "aromatic":  set("FYW"),
    "charged":   set("DEKRH"),
    "small":     set("AGSDT"),
    "tiny":      set("AGS"),
}

# GlobalDescriptor columns to retain
GLOBAL_COLS = ["Charge", "pI", "InstabilityInd", "AliphaticInd", "BomanInd", "HydrophRatio"]


# ---------------------------------------------------------------------------
# Grouped AAC  (9 features)
# ---------------------------------------------------------------------------
def _grouped_aac(seq: str) -> dict:
    n = len(seq)
    if n == 0:
        return {f"f_{name}": 0.0 for name in AA_GROUPS}
    return {f"f_{name}": sum(1 for aa in seq if aa in residues) / n
            for name, residues in AA_GROUPS.items()}


# ---------------------------------------------------------------------------
# FET local features  (3 features)
# ---------------------------------------------------------------------------
def _fet_local(seq: str) -> dict:
    """Relative position (D1) of first occurrence of a residue in each FET group."""
    n = len(seq)
    features = {}
    for name, residues in FET_GROUPS.items():
        pos = next((i for i, aa in enumerate(seq) if aa in residues), None)
        features[f"{name}_D1"] = (pos + 1) / n if (pos is not None and n > 0) else 0.0
    return features


# ---------------------------------------------------------------------------
# Solvent accessibility local features  (3 features)
# ---------------------------------------------------------------------------
def _sa_local(seq: str) -> dict:
    """Relative position of first occurrence of a residue in each SA group."""
    n = len(seq)
    features = {}
    for name, residues in SA_GROUPS.items():
        pos = next((i for i, aa in enumerate(seq) if aa in residues), None)
        features[f"{name}_D1"] = (pos + 1) / n if (pos is not None and n > 0) else 0.0
    return features


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def calculate_physicochemical_features(sequences: List[str], ids: List[str]) -> pd.DataFrame:
    """
    Calculate Macrel-inspired physicochemical features for a list of sequences.

    Features (22 total):
      - Global descriptors (6): Charge, pI, InstabilityInd, AliphaticInd, BomanInd, HydrophRatio
      - Hydrophobic moment (1): Eisenberg scale, angle=100 degrees (helix amphipathicity)
      - Grouped AAC (9):        fraction of residues in functional groups
      - FET local (3):          position of first residue in each FET group
      - Solvent access local (3): position of first residue in each SA group

    Args:
        sequences: list of amino acid sequences.
        ids: list of corresponding sequence identifiers.

    Returns:
        DataFrame with columns [ID, sequence, <22 features>].
    """
    if not sequences:
        return pd.DataFrame()

    # --- Global descriptors ---
    desc = GlobalDescriptor(sequences)
    desc.calculate_all(amide=True)
    global_df = pd.DataFrame(desc.descriptor, columns=desc.featurenames)[GLOBAL_COLS]

    # --- Hydrophobic moment (Eisenberg, 100 degrees) ---
    hm_values = []
    for seq in sequences:
        try:
            d = PeptideDescriptor([seq], scalename="eisenberg")
            d.calculate_moment(window=len(seq), angle=100, modality="max")
            hm_values.append(float(d.descriptor[0][0]))
        except Exception:
            hm_values.append(0.0)

    # --- Sequence-based features ---
    seq_rows = []
    for seq in sequences:
        row = {}
        row.update(_grouped_aac(seq))
        row.update(_fet_local(seq))
        row.update(_sa_local(seq))
        seq_rows.append(row)
    seq_df = pd.DataFrame(seq_rows)

    # --- Combine ---
    features_df = global_df.copy()
    features_df["HydrophobicMoment"] = hm_values
    features_df = pd.concat([features_df, seq_df], axis=1)
    features_df.insert(0, "ID", ids)
    features_df.insert(1, "sequence", sequences)

    return features_df
