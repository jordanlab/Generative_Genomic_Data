import allel
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import gc

def main():
    parser = argparse.ArgumentParser(
        description="Convert genomic data (VCF or CSV) to an unphased genotype tensor (CSV)"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input file (VCF or CSV)"
    )
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["vcf", "csv"],
        default="vcf",
        help="Type of input file: 'vcf' or 'csv' (default: vcf)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file. If not specified, auto-generated from input name."
    )
    parser.add_argument(
        "--samples-as-rows",
        action="store_true",
        help="For CSV input: if set, samples are rows and SNPs are columns. Otherwise, SNPs are rows and samples are columns."
    )
    args = parser.parse_args()
    
    input_path = args.input
    
    # Auto-generate output path if not provided
    if args.output is None:
        output_path = Path(input_path).stem + "_unphased_tensor.csv"
    else:
        output_path = args.output
    
    # Process based on input type
    if args.input_type == "vcf":
        # Load VCF (scikit-allel loads entire file; we minimize copies afterward)
        print(f"Loading VCF: {input_path}", flush=True)
        callset = allel.read_vcf(
            input_path,
            fields=['samples', 'calldata/GT', 'variants/CHROM', 'variants/POS', 'variants/REF', 'variants/ALT']
        )

        gt_array = callset['calldata/GT']  # shape: (n_variants, n_samples, 2), int8
        n_variants, n_samples, _ = gt_array.shape
        print(f"[INFO] Loaded {n_variants} variants × {n_samples} samples", flush=True)

        # Compute unphased genotypes directly with numpy (no torch / no pandas copy)
        print("[INFO] Computing unphased genotypes...", flush=True)
        allele0 = gt_array[:, :, 0]
        allele1 = gt_array[:, :, 1]
        unphased = np.where(
            (allele0 == -1) | (allele1 == -1), -1, allele0 + allele1
        ).astype(np.int8).T          # Transpose → (n_samples, n_variants)

        # Free the large genotype array as soon as possible
        del allele0, allele1, gt_array
        callset.pop('calldata/GT', None)
        gc.collect()

        # Build SNP-ID header row
        print("[INFO] Building SNP IDs...", flush=True)
        chroms   = callset['variants/CHROM']
        positions = callset['variants/POS'].astype(str)
        refs     = callset['variants/REF']
        alts     = callset['variants/ALT'][:, 0]
        snp_ids  = [f"chr{c}:{p}:{r}:{a}" for c, p, r, a in zip(chroms, positions, refs, alts)]

        del callset  # free remaining VCF metadata
        gc.collect()

        print(f"\nUnphased tensor shape: {unphased.shape} (samples × variants)", flush=True)
        print(f"First 5 samples, first 10 variants:")
        print(unphased[:5, :10])

        # Write CSV incrementally – one sample (row) at a time to avoid pandas overhead
        print(f"[INFO] Writing CSV incrementally to {output_path}...", flush=True)
        with open(output_path, 'w') as f:
            f.write(','.join(snp_ids) + '\n')
            for i in range(n_samples):
                f.write(','.join(map(str, unphased[i])) + '\n')
                if (i + 1) % 500 == 0:
                    print(f"[INFO] Written {i + 1}/{n_samples} samples", flush=True)
            print(f"[INFO] Written {n_samples}/{n_samples} samples", flush=True)

    elif args.input_type == "csv":
        # Load CSV
        print(f"Loading CSV: {input_path}")
        df = pd.read_csv(input_path)
        
        # If samples are rows and SNPs are columns, transpose so SNPs are rows and samples are columns
        if args.samples_as_rows:
            print("Transposing data: samples are rows, SNPs are columns")
            df = df.T
            df.reset_index(drop=True, inplace=True)
        
        print(f"Tensor shape: {df.shape}")
        print(f"First 5 rows (all columns):")
        print(df.iloc[:5, :10])
        
        # Save DataFrame to CSV
        df.to_csv(output_path, index=False)
    
    print(f"CSV file saved: {output_path}")

if __name__ == "__main__":
    main()