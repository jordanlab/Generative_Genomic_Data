import allel
import numpy as np
import torch

def load_vcf(file_path) -> dict:
    '''
    Converts a VCF file to an unphased genotype tensor.
    
    :param file_path: Path to the VCF file
    '''
    # Load VCF file using scikit-allel
    callset = allel.read_vcf(file_path, fields=['samples', 'calldata/GT', 'variants/CHROM', 'variants/POS', 'variants/REF', 'variants/ALT'])
    
    # Convert to a PyTorch tensor
    # genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
    
    return callset

def main():
    # Example usage
    file_path = '/home/sverma87/Generative_Genomic_Data/data/BAQ/BAQ_qc2_grch38_final.vcf'
    baq_vcf = load_vcf(file_path)
    
    baq_tensor = torch.tensor(baq_vcf['calldata/GT'], dtype=torch.float32)
    baq_samples = baq_vcf['samples']
    
    print(f"Loaded VCF file with {len(baq_samples)} samples and genotype tensor shape: {baq_tensor.shape}")
    
if __name__ == "__main__":
    main()