# amp_identifier/feature_extraction.py

import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# CTD property groupings (Chou & Shen 2007)
# Each property maps residues to one of three groups (1, 2, 3).
CTD_PROPERTIES = {
    "hydrophobicity":   {"1": "RKEDQN",   "2": "GASTPHY", "3": "CVLIMFW"},
    "volume":           {"1": "GASTC",    "2": "NDVEQIL", "3": "MHKFRYW"},
    "polarity":         {"1": "LIFWCMVY", "2": "PATGS",   "3": "HQRKNED"},
    "polarizability":   {"1": "GASDT",    "2": "CPNVEQIL","3": "KMHFRYW"},
    "charge":           {"1": "KR",       "2": "ANCQGHILMFPSTWYV", "3": "DE"},
    "secondary_struct": {"1": "EALMQKRH", "2": "VIYCWFT", "3": "GNPSD"},
    "solvent_access":   {"1": "ALFCGIVW", "2": "RKQEND",  "3": "MPSTHY"},
}


# ---------------------------------------------------------------------------
# AAC  (20 features)
# ---------------------------------------------------------------------------
def _aac(sequence: str) -> dict:
    """Amino acid composition: relative frequency of each of the 20 standard AAs."""
    n = len(sequence)
    if n == 0:
        return {f"AAC_{aa}": 0.0 for aa in AMINO_ACIDS}
    return {f"AAC_{aa}": sequence.count(aa) / n for aa in AMINO_ACIDS}


# ---------------------------------------------------------------------------
# DPC  (400 features)
# ---------------------------------------------------------------------------
def _dpc(sequence: str) -> dict:
    """Dipeptide composition: relative frequency of all 400 AA dipeptides."""
    pairs = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    n = len(sequence) - 1
    if n <= 0:
        return {f"DPC_{p}": 0.0 for p in pairs}
    counts = {}
    for i in range(n):
        dp = sequence[i:i+2]
        if len(dp) == 2 and dp[0] in AMINO_ACIDS and dp[1] in AMINO_ACIDS:
            counts[dp] = counts.get(dp, 0) + 1
    return {f"DPC_{p}": counts.get(p, 0) / n for p in pairs}


# ---------------------------------------------------------------------------
# CTD  (147 features: 7 properties x 21 values each)
# ---------------------------------------------------------------------------
def _residue_group_sequence(sequence: str, groups: dict) -> list:
    """Map each residue to its group label (1, 2, 3) or None if not found."""
    lookup = {}
    for label, residues in groups.items():
        for r in residues:
            lookup[r] = label
    return [lookup.get(aa) for aa in sequence]


def _ctd_single(sequence: str, prop_name: str, groups: dict) -> dict:
    """
    CTD for one property:
      - Composition (C): fraction in each group — 3 values
      - Transition (T): fraction of adjacent-pair transitions between groups — 3 values
      - Distribution (D): positions of 1st, 25%, 50%, 75%, 100% occurrence per group — 15 values
    Total: 21 values per property.
    """
    prefix = f"CTD_{prop_name}"
    group_seq = _residue_group_sequence(sequence, groups)
    n = len(group_seq)
    result = {}

    for g in ("1", "2", "3"):
        indices = [i for i, v in enumerate(group_seq) if v == g]
        cnt = len(indices)

        # Composition
        result[f"{prefix}_C{g}"] = cnt / n if n > 0 else 0.0

        # Distribution: 1st, 25%, 50%, 75%, last occurrence (as fraction of length)
        if cnt == 0:
            for q, label in zip([0, 25, 50, 75, 100], ["D1", "D25", "D50", "D75", "D100"]):
                result[f"{prefix}_{label}{g}"] = 0.0
        else:
            quantile_indices = [
                indices[0],
                indices[int(0.25 * cnt)],
                indices[int(0.50 * cnt)],
                indices[int(0.75 * cnt)],
                indices[-1],
            ]
            for idx, label in zip(quantile_indices, ["D1", "D25", "D50", "D75", "D100"]):
                result[f"{prefix}_{label}{g}"] = (idx + 1) / n if n > 0 else 0.0

    # Transition: fraction of consecutive pairs that switch between groups
    pair_counts = {"12": 0, "13": 0, "23": 0}
    n_pairs = 0
    for i in range(n - 1):
        g1, g2 = group_seq[i], group_seq[i + 1]
        if g1 is None or g2 is None:
            continue
        n_pairs += 1
        key = "".join(sorted([g1, g2]))
        if key in pair_counts:
            pair_counts[key] += 1
    denom = n_pairs if n_pairs > 0 else 1
    result[f"{prefix}_T12"] = pair_counts["12"] / denom
    result[f"{prefix}_T13"] = pair_counts["13"] / denom
    result[f"{prefix}_T23"] = pair_counts["23"] / denom

    return result


def _ctd(sequence: str) -> dict:
    """CTD features for all 7 physicochemical properties (147 features total)."""
    features = {}
    for prop_name, groups in CTD_PROPERTIES.items():
        features.update(_ctd_single(sequence, prop_name, groups))
    return features


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def calculate_physicochemical_features(sequences: List[str], ids: List[str]) -> pd.DataFrame:
    """
    Calculate physicochemical features for a list of sequences.

    Features included:
      - GlobalDescriptor (modlamp): MW, length, charge, hydrophobicity, pI, etc. (~15)
      - AAC: amino acid composition (20)
      - DPC: dipeptide composition (400)
      - CTD: composition/transition/distribution over 7 properties (147)

    Total: ~582 features per sequence.

    Args:
        sequences: list of amino acid sequences.
        ids: list of corresponding sequence identifiers.

    Returns:
        DataFrame with columns [ID, sequence, <features>].
    """
    if not sequences:
        return pd.DataFrame()

    # --- GlobalDescriptor features ---
    desc = GlobalDescriptor(sequences)
    desc.calculate_all(amide=True)
    global_df = pd.DataFrame(desc.descriptor, columns=desc.featurenames)

    # --- Sequence-based features ---
    seq_rows = []
    for seq in sequences:
        row = {}
        row.update(_aac(seq))
        row.update(_dpc(seq))
        row.update(_ctd(seq))
        seq_rows.append(row)
    seq_df = pd.DataFrame(seq_rows)

    # --- Combine ---
    features_df = pd.concat([global_df, seq_df], axis=1)
    features_df.insert(0, "ID", ids)
    features_df.insert(1, "sequence", sequences)

    return features_df
