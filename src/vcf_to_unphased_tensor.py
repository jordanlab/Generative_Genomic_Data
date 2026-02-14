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

    return callset

def create_unphased_tensor(callset: dict) -> torch.Tensor:
    '''
    Converts the genotype data from the callset to an unphased tensor.
    
    :param callset: Dictionary containing VCF data loaded by scikit-allel
    :return: Unphased genotype tensor of shape (variants, samples)
    '''
    # Extract genotype data and convert to a PyTorch tensor
    gt_array = callset['calldata/GT']  
    gt_tensor = torch.from_numpy(gt_array)  
    
    # Convert to unphased format by summing the two alleles for each sample and variant
    # Condition to handle missing data (-1) by setting the sum to -1 if either allele is missing
    # Else sum the two alleles to get the total count of alternate alleles (0, 1, or 2)
    unphased_tensor = torch.where(
        (gt_tensor[:, :, 0] == -1) | (gt_tensor[:, :, 1] == -1),
        -1.0,
        gt_tensor[:, :, 0] + gt_tensor[:, :, 1]
    )
    
    unphsdr_tensor = torch.transpose(unphased_tensor, 1, 0)  # Transpose to shape (variants, samples)
    
    return unphsdr_tensor

def main():
    # Example usage w/ BAQ Dataset
    file_path = '/home/sverma87/Generative_Genomic_Data/data/BAQ/BAQ_qc2_grch38_final.vcf'
    baq_vcf = load_vcf(file_path)
    baq_tensor = create_unphased_tensor(baq_vcf)
    
    print(f"\n2D Genotype tensor shape: {baq_tensor.shape}")
    print(f"First 5 samples (all variants) - 2D encoding:")
    print(baq_tensor[:5, :10])
    
if __name__ == "__main__":
    main()