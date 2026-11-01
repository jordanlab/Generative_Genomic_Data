"""Shared, read-only genotype value audit used by cWGAN and Yelmen pipelines.

Does not alter arrays. Keeps preprocessing decisions in the owning pipeline.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd


def audit(matrix, *, dataset, stage, path=None, missing_rule='negative', per_snp=False):
    """Return an auditable summary and (optionally) per-SNP missing/call counts.

    missing_rule='negative': cWGAN legacy negative-sentinel handling (NaN shown
        separately and is NOT treated as imputable by original cWGAN code).
    missing_rule='nan': Yelmen loader has already converted sentinels to NaN.
    For QC reporting, total missing is the union of nonfinite and negative calls.
    """
    a=np.asarray(matrix)
    if a.ndim != 2: raise ValueError('QC expects a sample-by-SNP matrix')
    finite=np.isfinite(a)
    negative=finite & (a<0)
    nan=np.isnan(a)
    inf=np.isinf(a)
    missing=~finite | negative
    observed=finite & ~negative
    fractions=observed & (a>=0) & (a<=2) & (a!=np.floor(a))
    outside=observed & ((a<0)|(a>2))
    total=int(a.size)
    missing_site=missing.sum(axis=0)
    counts={str(i):int(np.count_nonzero(a==i)) for i in range(3)}
    report={
       'dataset':str(dataset),'stage':str(stage),'source_path':str(path) if path else None,
       'individuals':int(a.shape[0]),'snps':int(a.shape[1]),'entries':total,
       'genotype_counts':counts,'missing_negative_1':int(np.count_nonzero(a==-1)),
       'missing_negative_9':int(np.count_nonzero(a==-9)),
       'missing_other_negative':int(np.count_nonzero(negative & (a!=-1) & (a!=-9))),
       'missing_nan':int(nan.sum()),'infinite':int(inf.sum()),
       'missing_total':int(missing.sum()),
       'missing_fraction':float(missing.sum()/total) if total else 0.,
       'fractional_dosages_0_to_2':int(fractions.sum()),
       'out_of_range_positive':int(outside.sum()),
       'snps_missing_ge_50pct':int((missing_site>=0.5*len(a)).sum()) if len(a) else 0,
       'snps_missing_ge_90pct':int((missing_site>=0.9*len(a)).sum()) if len(a) else 0,
       'snps_all_missing':int((missing_site==len(a)).sum()) if len(a) else 0,
       'missing_rule':missing_rule,
       'min_observed':float(np.min(a[observed])) if np.any(observed) else None,
       'max_observed':float(np.max(a[observed])) if np.any(observed) else None,
    }
    print('[GENOTYPE QC] {dataset} [{stage}]: {individuals:,} samples x {snps:,} SNPs; '
          '0/1/2={zero:,}/{one:,}/{two:,}; missing={missing_total:,} ({missing_fraction:.2%}), '
          'of which -1={missing_negative_1:,}, -9={missing_negative_9:,}, NaN={missing_nan:,}; '
          'fractional={fractional_dosages_0_to_2:,}; >2={out_of_range_positive:,}; '
          'SNPs >=50% missing={snps_missing_ge_50pct:,}'.format(
          **report,zero=counts['0'],one=counts['1'],two=counts['2']),flush=True)
    if report['missing_fraction']>0.1:
        print('[GENOTYPE QC WARNING] {dataset} [{stage}]: over 10% missing; imputation may dominate findings.'.format(**report),flush=True)
    if not counts['0'] and sum(counts.values()):
        print('[GENOTYPE QC WARNING] {dataset} [{stage}]: no observed genotype 0 calls; investigate VCF origin/coding.'.format(**report),flush=True)
    if report['infinite'] or report['out_of_range_positive']:
        print('[GENOTYPE QC WARNING] {dataset} [{stage}]: nonfinite/out-of-range calls require inspection.'.format(**report),flush=True)
    if missing_rule=='negative' and report['missing_nan']:
        print('[GENOTYPE QC WARNING] cWGAN legacy imputer only flags values <0; NaNs are not automatically imputed.',flush=True)
    return report


def per_snp_table(matrix,snps):
    a=np.asarray(matrix)
    if a.shape[1]!=len(snps):raise ValueError('SNP names must align with matrix')
    miss=~np.isfinite(a)|(a<0)
    obs=np.isfinite(a)&(a>=0)
    return pd.DataFrame({
        'snp':list(snps), 'n_missing':miss.sum(0),
        'missing_fraction':miss.mean(0) if len(a) else np.zeros(a.shape[1]),
        'n_0':(a==0).sum(0),'n_1':(a==1).sum(0),'n_2':(a==2).sum(0),
        'n_fractional_0_to_2':(obs&(a<=2)&(a!=np.floor(a))).sum(0),
        'n_outside_0_to_2':(obs&(a>2)).sum(0),
    })


def enforce(report,max_missing_fraction=None, fail_absent_zero=False):
    if max_missing_fraction is not None:
        if not 0<=max_missing_fraction<=1:raise ValueError('--max_missing_fraction must be within [0,1]')
        if report['missing_fraction']>max_missing_fraction:
            raise ValueError('{dataset} [{stage}]: missing fraction {missing_fraction:.2%} exceeds --max_missing_fraction {limit:.2%}'.format(
                **report,limit=max_missing_fraction))
    if fail_absent_zero and report['genotype_counts']['0']==0 and sum(report['genotype_counts'].values()):
        raise ValueError('{dataset} [{stage}]: no observed genotype-0 calls (--fail_on_absent_zero)'.format(**report))


def imputation_table(before, after, snps):
    """Describe realized per-SNP imputation on the actual selected training rows."""
    a=np.asarray(before);b=np.asarray(after)
    if a.shape!=b.shape or a.shape[1]!=len(snps):
        raise ValueError('Imputation audit matrices or SNP columns do not align')
    miss=~np.isfinite(a)|(a<0)
    fill_count=miss.sum(0)
    fill_value=np.full(a.shape[1],np.nan,dtype=np.float64)
    for j in np.flatnonzero(fill_count):
        vals=b[miss[:,j],j]
        vals=vals[np.isfinite(vals)]
        if vals.size:fill_value[j]=float(vals[0])
    return pd.DataFrame({'snp':list(snps),'n_missing_before':fill_count,
                         'n_filled_finite':(miss & np.isfinite(b) & (b>=0)).sum(0),
                         'fill_value_observed':fill_value,
                         'all_training_calls_missing':fill_count==len(a)})


def write_bundle(folder,reports,per_snp=None,metadata=None,imputations=None):
    """Write machine-readable QC audit, concise CSV, and detailed per-SNP CSVs."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    payload={'stages':reports,'metadata':metadata or {}}
    (folder/'input_qc_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    pd.json_normalize(reports,sep='_').to_csv(folder/'input_qc_summary.csv',index=False)
    for name,values in (per_snp or {}).items():
        matrix,snps=values
        per_snp_table(matrix,snps).to_csv(folder/f'{name}_per_snp.csv',index=False)
    for name,values in (imputations or {}).items():
        before,after,snps=values
        imputation_table(before,after,snps).to_csv(folder/f'{name}_imputed_snps.csv',index=False)
    print(f'[GENOTYPE QC] Report: {folder / "input_qc_report.json"}',flush=True)
    return folder/'input_qc_report.json'


def discrete_real_for_evaluation(real,holdout=None):
    """Use the same observed-hard-call per-SNP modal rule as the cWGAN evaluator.

    This is a paper-style evaluation representation, NOT the training matrix.
    """
    a=np.asarray(real,dtype=np.float32)
    observed=np.isfinite(a)&np.isin(a,(0,1,2))
    mode=np.array([np.bincount(a[observed[:,j],j].astype(np.int64),minlength=3).argmax()
                   if observed[:,j].any() else 0 for j in range(a.shape[1])],dtype=np.float32)
    def convert(x):
        if x is None:return None,0
        values=np.asarray(x,dtype=np.float32).copy()
        invalid=~np.isin(values,(0,1,2))
        rr,cc=np.where(invalid)
        values[rr,cc]=mode[cc]
        return values.astype(np.int8),int(invalid.sum())
    real_discrete,n_real=convert(real)
    other_discrete,n_other=convert(holdout)
    return real_discrete,other_discrete,{'real':n_real,'holdout':n_other}
