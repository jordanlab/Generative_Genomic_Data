#!/usr/bin/env python3
"""
Convert a dosage CSV (rows=samples, cols=variants, values in {0,1,2,-1}) to PLINK files.

Expected column format:
    chr:pos:ref:alt
Example:
    chr1:2379954:A:G

Pipeline:
    CSV -> VCF -> PLINK BED/BIM/FAM (or PGEN if requested)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

import pandas as pd

VARIANT_RE = re.compile(r"^(?:chr)?([^:]+):(\d+):([ACGTN]+):([ACGTN]+)$", re.IGNORECASE)
GT_MAP = {
    -1: "./.",
    0: "0/0",
    1: "0/1",
    2: "1/1",
}


def parse_variant(col_name: str) -> tuple[str, str, str, str, str]:
    """
    Parse variant of form chr:pos:ref:alt.
    Returns (chrom, pos, vid, ref, alt).
    """
    m = VARIANT_RE.match(col_name.strip())
    if not m:
        raise ValueError(
            f"Column '{col_name}' is not in expected format 'chr:pos:ref:alt'"
        )

    chrom, pos, ref, alt = m.groups()
    chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    vid = f"{chrom}:{pos}:{ref}:{alt}"
    return chrom, pos, vid, ref.upper(), alt.upper()


def dosage_to_gt(value) -> str:
    try:
        v = int(value)
    except Exception:
        return "./."
    return GT_MAP.get(v, "./.")


def csv_to_vcf(input_csv: Path, output_vcf: Path, sample_prefix: str = "S") -> int:
    """Create a diploid GT-only VCF from dosage CSV."""
    df = pd.read_csv(input_csv)
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Input CSV is empty.")

    variant_info = [parse_variant(c) for c in df.columns]
    sample_ids = [f"{sample_prefix}{i+1}" for i in range(df.shape[0])]

    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    with output_vcf.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")

        # Minimal VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        writer.writerow(
            [
                "#CHROM",
                "POS",
                "ID",
                "REF",
                "ALT",
                "QUAL",
                "FILTER",
                "INFO",
                "FORMAT",
                *sample_ids,
            ]
        )

        # One row per variant
        for j, (chrom, pos, vid, ref, alt) in enumerate(variant_info):
            dosages = df.iloc[:, j].tolist()
            gt_calls = [dosage_to_gt(v) for v in dosages]
            writer.writerow([chrom, pos, vid, ref, alt, ".", "PASS", ".", "GT", *gt_calls])

    return df.shape[0]


def run_plink(output_vcf: Path, out_prefix: Path, make_pgen: bool = False) -> None:
    cmd = [
        "plink2",
        "--vcf",
        str(output_vcf),
        "--allow-extra-chr",
        "--double-id",
    ]
    if make_pgen:
        cmd.extend(["--make-pgen"])
    else:
        cmd.extend(["--make-bed"])
    cmd.extend(["--out", str(out_prefix)])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert dosage CSV to PLINK files via VCF")
    p.add_argument("--input-csv", required=True, type=Path, help="Input dosage CSV")
    p.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output prefix for PLINK files (no extension)",
    )
    p.add_argument(
        "--temp-vcf",
        type=Path,
        default=None,
        help="Optional temp VCF path. Default: <output-prefix>.vcf",
    )
    p.add_argument(
        "--sample-prefix",
        default="S",
        help="Prefix for generated sample IDs (default: S)",
    )
    p.add_argument(
        "--make-pgen",
        action="store_true",
        help="Write PGEN/PVAR/PSAM instead of BED/BIM/FAM",
    )
    p.add_argument(
        "--skip-plink",
        action="store_true",
        help="Only write VCF, do not run plink2",
    )
    args = p.parse_args()

    temp_vcf = args.temp_vcf if args.temp_vcf else args.output_prefix.with_suffix(".vcf")

    n_samples = csv_to_vcf(args.input_csv, temp_vcf, args.sample_prefix)
    print(f"Wrote VCF: {temp_vcf} ({n_samples} samples)")

    if not args.skip_plink:
        run_plink(temp_vcf, args.output_prefix, make_pgen=args.make_pgen)
        if args.make_pgen:
            print("Done. Wrote .pgen/.pvar/.psam files.")
        else:
            print("Done. Wrote .bed/.bim/.fam files.")


if __name__ == "__main__":
    main()
