"""Optional LOGAN-style membership attacks for archived conditional WGAN checkpoints.

Unlike ordinary fidelity metrics, a membership attack requires explicitly supplied
member subjects, non-member subjects that were excluded from *all* model fitting,
and their matching per-individual ancestry condition vectors. If these provenance
requirements are not met, DO NOT interpret the attacks as privacy evidence.

Example:
 python src/run_membership_attacks.py --model_dir RUN/BAQ --weight best \
   --train_input members.csv --test_input strictly_excluded.csv \
   --train_conditions members_ancestry.csv --test_conditions test_ancestry.csv \
   --blackbox_epochs 100
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_saved_genotypes import load_checkpoint_model
from genotype_membership_attacks import whitebox_attack, blackbox_surrogate_attack


def _genotypes(path,snps):
    frame=pd.read_csv(path)
    if not all(x in frame.columns for x in snps):
        raise ValueError(f'{path}: SNP columns do not match checkpoint SNPs')
    a=frame.loc[:,snps].to_numpy(float)
    if not np.all(np.isfinite(a) & np.isin(a,[0,1,2])):
        raise ValueError(f'{path}: expected nonmissing discrete 0,1,2 dosages')
    return a.astype(np.int8)


def _conditions(path,n):
    df=pd.read_csv(path)
    if set(['AFR','EUR','AMR']).issubset(df.columns):
        df=df[['AFR','EUR','AMR']]
    if df.shape != (n,3):
        raise ValueError(f'{path}: expected {n} rows x 3 ancestry columns')
    x=df.to_numpy(np.float32)
    if not np.all(np.isfinite(x)) or np.any(x<0) or not np.allclose(x.sum(1),1,atol=1e-3):
        raise ValueError(f'{path}: ancestry conditions must be finite nonnegative rows summing to 1')
    return x


def run_attacks(model_dir,weight,train_input,test_input,train_conditions,test_conditions,
                synthetic_input=None,synthetic_conditions=None,output_dir=None,
                blackbox_epochs=0,seed=42,device='auto'):
    model_dir=Path(model_dir)
    ck=model_dir/'checkpoints'/f'checkpoint_{weight}.pt'
    model,backend,snps,mean,components,state=load_checkpoint_model(
        ck,model_dir/'checkpoints'/'pca_model.npz',device)
    real=_genotypes(train_input,snps);test=_genotypes(test_input,snps)
    cr=_conditions(train_conditions,len(real));ct=_conditions(test_conditions,len(test))
    if output_dir is None:output_dir=model_dir/'membership_attacks'/weight
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    result={'checkpoint':str(ck),'status':'member_vs_explicit_excluded_test',
            'train_n':len(real),'test_n':len(test),'weight':weight,
            'important':'User must verify train subjects were included and test subjects excluded from fitting, preprocessing, reference PCA and model selection; no DP guarantee.',
            'whitebox':whitebox_attack(model,backend,real,test,cr,ct,components,mean,output_dir)}
    if blackbox_epochs>0:
        if not synthetic_input or not synthetic_conditions:
            raise ValueError('Blackbox attack requires --synthetic_input and --synthetic_conditions')
        syn=_genotypes(synthetic_input,snps);cs=_conditions(synthetic_conditions,len(syn))
        result['blackbox_surrogate']=blackbox_surrogate_attack(real,test,cr,ct,
            syn,cs,components,mean,output_dir,blackbox_epochs,seed,device)
    else:
        result['blackbox_surrogate']={'status':'not_run','reason':'Set --blackbox_epochs plus synthetic files'}
    (output_dir/'membership_attack_summary.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for arg in ('model_dir','train_input','test_input','train_conditions','test_conditions'):
        ap.add_argument('--'+arg,required=True)
    ap.add_argument('--weight',default='best',choices=['baseline','first','best','last'])
    ap.add_argument('--synthetic_input')
    ap.add_argument('--synthetic_conditions')
    ap.add_argument('--output_dir')
    ap.add_argument('--blackbox_epochs',type=int,default=0)
    ap.add_argument('--device',default='auto');ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    print(json.dumps(run_attacks(**vars(args)),indent=2))

if __name__=='__main__':main()
