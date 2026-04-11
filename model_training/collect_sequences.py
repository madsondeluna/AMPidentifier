# model_training/collect_sequences.py
#
# Download AMP (positive) and non-AMP (negative) sequences from public databases,
# merge with existing FASTA files, and apply CD-HIT to remove redundancy.
#
# Sources:
#   Positive:  DBAASP v3 API, APD3 (direct download), DRAMP 3.0 (direct download)
#   Negative:  UniProt REST API (reviewed, non-AMP, length 10-100 aa)
#
# Output:
#   model_training/data/positive_sequences.fasta   (deduplicated, replaces existing)
#   model_training/data/negative_sequences.fasta   (deduplicated, replaces existing)
#
# Run from project root:
#   python -m model_training.collect_sequences

import os
import re
import ssl
import time
import shutil
import subprocess
import urllib.request
import urllib.parse

# SSL context that bypasses certificate verification (used only for APD3,
# which has a misconfigured certificate on the server side).
_SSL_NO_VERIFY = ssl.create_default_context()
_SSL_NO_VERIFY.check_hostname = False
_SSL_NO_VERIFY.verify_mode = ssl.CERT_NONE

DATA_DIR    = "model_training/data"
CACHE_DIR   = os.path.join(DATA_DIR, "raw_downloads")
CDHIT_ID    = 0.90   # sequence identity threshold
CDHIT_N     = 5      # word size for 0.90 threshold (CD-HIT recommendation)
MIN_LEN     = 10
MAX_LEN     = 200

POSITIVE_OUT = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_OUT = os.path.join(DATA_DIR, "negative_sequences.fasta")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_url(url: str, dest: str, retries: int = 3, delay: float = 2.0,
               ssl_context=None):
    """Download url to dest with retry logic."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"    GET {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60,
                                        context=ssl_context) as resp, \
                 open(dest, "wb") as f:
                f.write(resp.read())
            return
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _parse_fasta(path: str) -> dict:
    """Return {header: sequence} from a FASTA file. Skips empty sequences."""
    seqs = {}
    header = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and buf:
                    seqs[header] = "".join(buf)
                header = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line.upper())
    if header and buf:
        seqs[header] = "".join(buf)
    return seqs


def _write_fasta(seqs: dict, path: str):
    with open(path, "w") as f:
        for header, seq in seqs.items():
            f.write(f">{header}\n{seq}\n")
    print(f"  Written: {path}  ({len(seqs)} sequences)")


def _filter_length(seqs: dict, min_len: int, max_len: int) -> dict:
    filtered = {h: s for h, s in seqs.items() if min_len <= len(s) <= max_len}
    removed = len(seqs) - len(filtered)
    if removed:
        print(f"  Length filter ({min_len}-{max_len} aa): removed {removed}, kept {len(filtered)}")
    return filtered


def _filter_nonstandard(seqs: dict) -> dict:
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    filtered = {h: s for h, s in seqs.items()
                if all(aa in standard for aa in s)}
    removed = len(seqs) - len(filtered)
    if removed:
        print(f"  Non-standard AA filter: removed {removed}, kept {len(filtered)}")
    return filtered


def _run_cdhit(input_fasta: str, output_fasta: str):
    """Run cd-hit on input_fasta, write to output_fasta."""
    cdhit_bin = shutil.which("cd-hit")
    if not cdhit_bin:
        raise RuntimeError("cd-hit not found in PATH. Install with: brew install cd-hit")
    cmd = [
        cdhit_bin,
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(CDHIT_ID),
        "-n", str(CDHIT_N),
        "-M",  "4000",   # memory limit MB
        "-T",  "0",      # use all CPU threads
        "-d",  "0",      # full header in .clstr
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("cd-hit failed")
    # Count output
    n = sum(1 for line in open(output_fasta) if line.startswith(">"))
    print(f"  CD-HIT output: {n} representative sequences")


# ---------------------------------------------------------------------------
# Positive sources
# ---------------------------------------------------------------------------
def _download_apd3() -> dict:
    """APD3 2024 — natural AMPs FASTA (current release).

    aps.unmc.edu has a misconfigured TLS certificate; SSL verification is
    intentionally bypassed for this download only.
    """
    cache = os.path.join(CACHE_DIR, "apd3_natural_2024.fasta")
    if not os.path.exists(cache):
        url = "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta"
        _fetch_url(url, cache, ssl_context=_SSL_NO_VERIFY)
    seqs = _parse_fasta(cache)
    renamed = {f"APD3_{h}": s for h, s in seqs.items()}
    print(f"  APD3: {len(renamed)} sequences parsed")
    return renamed


def _download_campr3() -> dict:
    """CAMPR3 — full FASTA via fasta.php endpoint (SSL bypass required)."""
    cache = os.path.join(CACHE_DIR, "campr3.fasta")
    if not os.path.exists(cache):
        url = "https://www.camp3.bicnirrh.res.in/fasta.php"
        _fetch_url(url, cache, ssl_context=_SSL_NO_VERIFY)
    # The file uses <br> as line separator instead of newlines
    with open(cache) as f:
        raw = f.read()
    raw = re.sub(r"<br\s*/?>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    # Write clean FASTA to temp file and parse
    clean_path = cache + ".clean.fasta"
    with open(clean_path, "w") as f:
        f.write(raw)
    seqs = _parse_fasta(clean_path)
    renamed = {f"CAMPR3_{h}": s for h, s in seqs.items()}
    print(f"  CAMPR3: {len(renamed)} sequences parsed")
    return renamed


def _download_dramp() -> dict:
    """DRAMP 3.0 — general AMP FASTA download."""
    cache = os.path.join(CACHE_DIR, "dramp_general.fasta")
    if not os.path.exists(cache):
        url = "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.fasta"
        _fetch_url(url, cache)
    seqs = _parse_fasta(cache)
    renamed = {f"DRAMP_{h}": s for h, s in seqs.items()}
    print(f"  DRAMP: {len(renamed)} sequences parsed")
    return renamed


def collect_positives() -> dict:
    print("\n=== Collecting positive sequences (AMPs) ===")
    existing = _parse_fasta(POSITIVE_OUT)
    print(f"  Existing positives: {len(existing)}")

    # Rename existing to avoid header collision
    existing_renamed = {f"existing_pos_{i}": s for i, s in enumerate(existing.values())}

    all_seqs = {**existing_renamed}

    try:
        all_seqs.update(_download_apd3())
    except Exception as e:
        print(f"  APD3 download failed: {e}")

    try:
        all_seqs.update(_download_campr3())
    except Exception as e:
        print(f"  CAMPR3 download failed: {e}")

    try:
        all_seqs.update(_download_dramp())
    except Exception as e:
        print(f"  DRAMP download failed: {e}")

    print(f"  Total before filtering: {len(all_seqs)}")
    all_seqs = _filter_length(all_seqs, MIN_LEN, MAX_LEN)
    all_seqs = _filter_nonstandard(all_seqs)
    return all_seqs


# ---------------------------------------------------------------------------
# Negative source
# ---------------------------------------------------------------------------
def _download_uniprot_negatives() -> dict:
    """UniProt REST API — reviewed sequences without antimicrobial annotation.

    Query: reviewed=true, length 10-200 aa, NOT keyword 'Antimicrobial'.
    Returns up to 50000 sequences.
    """
    cache = os.path.join(CACHE_DIR, "uniprot_negatives.fasta")
    if not os.path.exists(cache):
        query = (
            "(reviewed:true) "
            "AND (length:[10 TO 200]) "
            "NOT (keyword:KW-0929) "          # KW-0929 = Antimicrobial
            "NOT (keyword:KW-0044) "          # KW-0044 = Antibiotic
        )
        encoded = urllib.parse.quote(query)
        url = (
            f"https://rest.uniprot.org/uniprotkb/stream"
            f"?query={encoded}&format=fasta&size=50000"
        )
        _fetch_url(url, cache)
    seqs = _parse_fasta(cache)
    renamed = {f"UniProt_{h}": s for h, s in seqs.items()}
    print(f"  UniProt negatives: {len(renamed)} sequences parsed")
    return renamed


def collect_negatives() -> dict:
    print("\n=== Collecting negative sequences (non-AMPs) ===")
    existing = _parse_fasta(NEGATIVE_OUT)
    print(f"  Existing negatives: {len(existing)}")

    existing_renamed = {f"existing_neg_{i}": s for i, s in enumerate(existing.values())}
    all_seqs = {**existing_renamed}

    try:
        all_seqs.update(_download_uniprot_negatives())
    except Exception as e:
        print(f"  UniProt download failed: {e}")

    print(f"  Total before filtering: {len(all_seqs)}")
    all_seqs = _filter_length(all_seqs, MIN_LEN, MAX_LEN)
    all_seqs = _filter_nonstandard(all_seqs)
    return all_seqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Collect and filter
    pos_seqs = collect_positives()
    neg_seqs = collect_negatives()

    # Write merged FASTAs for CD-HIT input
    merged_pos = os.path.join(CACHE_DIR, "merged_positives.fasta")
    merged_neg = os.path.join(CACHE_DIR, "merged_negatives.fasta")
    _write_fasta(pos_seqs, merged_pos)
    _write_fasta(neg_seqs, merged_neg)

    # Run CD-HIT
    print(f"\n=== CD-HIT deduplication (identity >= {CDHIT_ID}) ===")
    dedup_pos = os.path.join(CACHE_DIR, "dedup_positives.fasta")
    dedup_neg = os.path.join(CACHE_DIR, "dedup_negatives.fasta")

    print("  Deduplicating positives...")
    _run_cdhit(merged_pos, dedup_pos)

    print("  Deduplicating negatives...")
    _run_cdhit(merged_neg, dedup_neg)

    # Balance dataset
    pos_final = _parse_fasta(dedup_pos)
    neg_final = _parse_fasta(dedup_neg)
    n = min(len(pos_final), len(neg_final))

    import random
    random.seed(42)
    pos_keys = random.sample(list(pos_final.keys()), n)
    neg_keys = random.sample(list(neg_final.keys()), n)
    pos_balanced = {k: pos_final[k] for k in pos_keys}
    neg_balanced = {k: neg_final[k] for k in neg_keys}

    # Rename to simple sequential IDs for consistency
    pos_out = {f"AMP_{i+1}": s for i, s in enumerate(pos_balanced.values())}
    neg_out = {f"non_amp_{i+1}": s for i, s in enumerate(neg_balanced.values())}

    # Overwrite final FASTAs
    print(f"\n=== Final dataset: {n} AMPs + {n} non-AMPs ===")
    _write_fasta(pos_out, POSITIVE_OUT)
    _write_fasta(neg_out, NEGATIVE_OUT)

    print("\nDone. Re-run feature_analysis.py and train.py to update the model.")


if __name__ == "__main__":
    main()
