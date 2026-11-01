"""Unphased-dosage adaptations of Yelmen et al. (2023) figures and supplements.

All LD pairs are intrachromosomal.  'all_chromosomes' pools such pairs; it
NEVER computes a biological LD-decay curve from cross-chromosome pairs.
Haplotype-only statistics are explicitly labelled as genotype surrogates.
"""
from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance, gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def _corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) < 1e-12 or np.std(b[keep]) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def _json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(str(type(obj)))


def _clean_geno(a, label):
    """Evaluation never silently converts out-of-range synthetic genotypes."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f'{label}: require a finite n_samples x n_snps matrix; impute missing explicitly')
    if not np.all((a == 0) | (a == 1) | (a == 2)):
        raise ValueError(f'{label}: require discrete genotypes 0, 1, 2; not imputed dosages')
    return a.astype(np.int8)


def chromosome_positions(snps):
    """Parse chr10:706253:T:C and similar; nonparseable positions use file order."""
    chrom, bp, fallback = [], [], []
    for i, snp in enumerate(snps):
        m = re.match(r'^(?:chr)?(\d+|X|Y|MT|M):([0-9]+)(?::|$)', str(snp), re.I)
        chrom.append(('chr' + m.group(1).upper().replace('M', 'MT')) if m else 'unknown')
        bp.append(int(m.group(2)) if m else i + 1)
        fallback.append(m is None)
    return np.array(chrom), np.array(bp, dtype=np.int64), bool(any(fallback))


def plot_xy(x, y, dest, xlabel, ylabel, title, lim=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(x, y, s=5, alpha=.25, rasterized=True)
    if lim is not None: ax.plot([lim[0], lim[1]], [lim[0], lim[1]], '--', color='black', lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    text = f'r={_corr(x,y):.3f}'
    if len(x) >= 3 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        text += f'  slope={slope:.3f}  intercept={intercept:.3f}'
    ax.text(.02, .98, text, va='top', transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(dest, dpi=155); plt.close(fig)


def frequency_statistics(real, syn, plots):
    fr, fs = real.mean(0)/2, syn.mean(0)/2
    low = fr <= .2
    plot_xy(fr, fs, plots/'allele_frequency.png', 'Real alt allele frequency',
            'Synthetic alt allele frequency', 'Allele frequency: all sites', (0, 1))
    if low.sum() > 2:
        plot_xy(fr[low], fs[low], plots/'allele_frequency_low.png',
                'Real alt allele frequency (<=0.2)', 'Synthetic alt allele frequency',
                'Low-frequency allele preservation', (0, .2))
    fixed = (fs == 0) | (fs == 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fr[fixed], bins=30, alpha=.8)
    ax.set(xlabel='Real alternative-allele frequency', ylabel='Count',
           title='Real frequency at synthetic-fixed SNPs')
    fig.tight_layout(); fig.savefig(plots/'fixed_allele_real_frequencies.png', dpi=155); plt.close(fig)
    return {'af_r': _corr(fr, fs), 'af_low_frequency_r': _corr(fr[low], fs[low]),
            'af_mae': float(np.mean(abs(fr - fs))),
            'synthetic_fixed_snps': int(fixed.sum()),
            'fixed_in_synthetic_not_real': int(np.sum(fixed & (fr > 0) & (fr < 1)))}


def genotype_frequency_stats(real, syn, plots):
    het_r = (real==1).mean(1); het_s = (syn==1).mean(1)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(het_r, bins=30, alpha=.6, density=True, label='real')
    ax.hist(het_s, bins=30, alpha=.6, density=True, label='synthetic')
    ax.legend(); ax.set(xlabel='Heterozygote fraction per individual', ylabel='Density')
    fig.tight_layout(); fig.savefig(plots/'heterozygosity.png', dpi=155); plt.close(fig)
    # Diploid-specific diagnostic absent in the phased paper.
    fr = real.mean(0)/2; fs = syn.mean(0)/2
    hetr = (real==1).mean(0); hets = (syn==1).mean(0)
    hwer = hetr - 2*fr*(1-fr); hwes = hets - 2*fs*(1-fs)
    plot_xy(hwer, hwes, plots/'hardy_weinberg_deviation.png',
            'Real observed minus expected heterozygosity',
            'Synthetic observed minus expected heterozygosity',
            'Per-SNP HWE heterozygosity residual')
    return {'heterozygosity_real_mean': float(het_r.mean()),
            'heterozygosity_synthetic_mean': float(het_s.mean()),
            'heterozygosity_mae': float(abs(het_r.mean()-het_s.mean())),
            'hwe_heterozygosity_residual_r': _corr(hwer, hwes),
            'genotype_class_frequencies_real': [float(np.mean(real==v)) for v in range(3)],
            'genotype_class_frequencies_synthetic': [float(np.mean(syn==v)) for v in range(3)]}


def _genotype_r2(g, a, b, chunk=2048):
    out = np.full(len(a), np.nan)
    for start in range(0, len(a), chunk):
        sl = slice(start, start+chunk)
        x, y = g[:, a[sl]].astype(float), g[:, b[sl]].astype(float)
        x -= x.mean(0); y -= y.mean(0)
        den = np.sqrt((x*x).sum(0) * (y*y).sum(0))
        r = np.divide((x*y).sum(0), den, out=np.full(x.shape[1], np.nan), where=den>0)
        out[sl] = r*r
    return out


def _ld_pairs(indices, bp, max_pairs, rng):
    """Log-distance-stratified intrachromosomal pairs; exact positions if available."""
    indices = np.asarray(indices)
    if len(indices) < 2: return np.array([], int), np.array([], int)
    order = np.argsort(bp[indices], kind='stable')
    ids, pos = indices[order], bp[indices][order]
    span = max(1, int(pos[-1] - pos[0]))
    # Draw distance bins uniformly in log space; locate target index by searchsorted.
    # Fallback for duplicate positions and position-free files uses index distances.
    a, b = [], []
    for _ in range(max_pairs * 3):
        if len(a) >= max_pairs: break
        j = int(rng.integers(0, len(ids)-1))
        target = pos[j] + np.exp(rng.uniform(0, np.log(span+1)))
        k = int(np.searchsorted(pos, target, side='left'))
        if k <= j: k = j+1
        if k >= len(ids): k = len(ids)-1
        if k != j:
            a.append(ids[j]); b.append(ids[k])
    if not a: return np.array([],int),np.array([],int)
    pairs = np.unique(np.sort(np.column_stack([a,b]),axis=1),axis=0)
    return pairs[:,0], pairs[:,1]


def ld_summary(real, syn, snps, base, max_pairs=3000, seed=42,
               heatmap_snps=96, bins=12):
    chrom, bp, fallback = chromosome_positions(snps)
    rng = np.random.default_rng(seed)
    scopes = {'all_chromosomes': np.arange(len(snps))}
    for ch in sorted(set(chrom), key=lambda c: (c=='unknown',c)):
        scopes[ch] = np.where(chrom == ch)[0]
    metrics = {}
    for scope, indices in scopes.items():
        ddf = pd.DataFrame()  # no stale bins across chromosomes
        scope_dir = base / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        if scope == 'all_chromosomes':
            a, b = [], []
            viable = [(c, ii) for c,ii in scopes.items() if c!='all_chromosomes' and len(ii)>1]
            weights = np.array([len(ii)-1 for _,ii in viable], float)
            if len(weights):
                counts = rng.multinomial(max_pairs, weights/weights.sum())
                for (_, ii), count in zip(viable,counts):
                    ai, bi = _ld_pairs(ii,bp,max(int(count),1),rng)
                    a.extend(ai); b.extend(bi)
            a,b=np.array(a,int),np.array(b,int)
        else: a,b = _ld_pairs(indices,bp,max_pairs,rng)
        if len(a) == 0:
            metrics[scope] = {'pairs_valid':0,'note':'fewer than two SNPs'}
            continue
        dr = _genotype_r2(real,a,b); ds = _genotype_r2(syn,a,b)
        mask = np.isfinite(dr)&np.isfinite(ds)
        a,b,dr,ds=a[mask],b[mask],dr[mask],ds[mask]
        delta=np.abs(bp[b]-bp[a]).astype(float)
        if not len(a):
            metrics[scope]={'pairs_valid':0,'note':'no polymorphic pairs common to real and synthetic'}
            continue
        # Use integer site gap only when position cannot be recovered.
        if fallback and (scope=='unknown' or scope=='all_chromosomes'):
            unknown=(chrom[a]=='unknown')
            delta[unknown] = abs(b[unknown]-a[unknown])
        df=pd.DataFrame({'snp1': np.asarray(snps)[a], 'snp2':np.asarray(snps)[b],
                         'distance': delta, 'r2_real':dr, 'r2_synthetic':ds})
        df.to_csv(scope_dir/'ld_pairs.csv',index=False)
        plot_xy(dr,ds,scope_dir/'ld_r2_scatter.png','Real dosage $r^2$',
                'Synthetic dosage $r^2$',f'LD pair fidelity: {scope}',(0,1))
        positive = delta>0
        if positive.sum() >= 2:
            edges=np.geomspace(max(float(delta[positive].min()),1.),max(float(delta.max()),1.)+1, bins+1)
            edges=np.unique(edges)
            records=[]
            for lo, hi in zip(edges[:-1],edges[1:]):
                sel=(delta >= lo) & (delta < hi)
                if sel.sum():
                    records.append({'distance_center':float(np.sqrt(lo*hi)),
                                    'count':int(sel.sum()),'mean_real_r2':float(dr[sel].mean()),
                                    'mean_synthetic_r2':float(ds[sel].mean())})
            ddf=pd.DataFrame(records)
            ddf.to_csv(scope_dir/'ld_decay.csv',index=False)
            if len(ddf):
                fig,ax=plt.subplots(figsize=(7,4)); ax.plot(ddf.distance_center,ddf.mean_real_r2,'o-',label='real')
                ax.plot(ddf.distance_center,ddf.mean_synthetic_r2,'o-',label='synthetic')
                ax.set_xscale('log'); ax.legend(); ax.set(xlabel='Physical bp separation (index separation if unavailable)',
                                  ylabel='Mean dosage $r^2$',title=f'LD decay: {scope}')
                fig.tight_layout();fig.savefig(scope_dir/'ld_decay.png',dpi=160);plt.close(fig)
                plot_xy(ddf.mean_real_r2.to_numpy(),ddf.mean_synthetic_r2.to_numpy(),
                        scope_dir/'ld_decay_correspondence.png','Real binned $r^2$',
                        'Synthetic binned $r^2$', 'LD decay bin correspondence', (0,1))
        # Full LD heatmap is at most 96 sites for memory safety; first chromosome
        # for pooled run, but plot labelled with that region; ordered by basepair.
        heat_ids = indices if scope!='all_chromosomes' else next((ii for ch,ii in scopes.items() if ch!='all_chromosomes' and len(ii)>=2), np.array([],int))
        if len(heat_ids)>=2:
            heat_ids=heat_ids[np.argsort(bp[heat_ids],kind='stable')][:heatmap_snps]
            ai,bi=np.triu_indices(len(heat_ids),k=1)
            real_heat=_genotype_r2(real,heat_ids[ai],heat_ids[bi]); syn_heat=_genotype_r2(syn,heat_ids[ai],heat_ids[bi])
            heat=np.full((len(heat_ids),len(heat_ids)),np.nan)
            heat[bi,ai]=real_heat;heat[ai,bi]=syn_heat
            np.fill_diagonal(heat,1)
            fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(heat,vmin=0,vmax=1,origin='lower',cmap='viridis')
            ax.set(title='Lower: real; upper: synthetic (dosage $r^2$)',xlabel='Ordered SNP index',ylabel='Ordered SNP index')
            fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(scope_dir/'ld_heatmap.png',dpi=150);plt.close(fig)
        metrics[scope]={'pairs_valid':int(len(dr)), 'pair_r2_r':_corr(dr,ds),
                        'ld_decay_radar_score': (float(1 - np.sum((ddf.mean_real_r2.to_numpy() - ddf.mean_synthetic_r2.to_numpy())**2) / np.sum(ddf.mean_real_r2.to_numpy()**2)) if 'ddf' in locals() and len(ddf) and np.sum(ddf.mean_real_r2.to_numpy()**2) > 0 else float('nan')),
                        'pair_r2_mae':float(abs(dr-ds).mean()),
                        'mean_real_r2':float(dr.mean()),'mean_synthetic_r2':float(ds.mean())}
    return metrics, fallback


def triplet_summary(real, syn, snps, out, n_triplets=2000, seed=42):
    chrom,bp,_ = chromosome_positions(snps)
    rng=np.random.default_rng(seed)
    # Standardized central *third* cross-moment, which is not the phased
    # haplotype three-point correlation. Use the same triplets in both cohorts.
    def moments(g, tr):
        mu=g.mean(0);sd=g.std(0)
        out=[]
        for st in range(0,len(tr),512):
            t=tr[st:st+512]
            if not len(t):continue
            z=np.stack([(g[:,t[:,i]]-mu[t[:,i]])/np.maximum(sd[t[:,i]],1e-8) for i in range(3)])
            vals=np.mean(z[0]*z[1]*z[2],axis=0)
            vals[np.any(sd[t]<1e-8,axis=1)]=np.nan
            out.extend(vals)
        return np.asarray(out)
    records=[]
    for step in (1,4,16,64,256,512,1024,'random'):
        triples=[]
        for ch in np.unique(chrom):
            ind=np.where(chrom==ch)[0]; ind=ind[np.argsort(bp[ind],kind='stable')]
            if len(ind)<3: continue
            if step!='random':
                slots=np.arange(0,len(ind)-2*step)
                if len(slots):
                    pick=rng.choice(slots,size=min(len(slots),max(1,n_triplets//max(len(np.unique(chrom)),1))),replace=False)
                    triples.extend(np.stack([ind[pick],ind[pick+step],ind[pick+2*step]],1).tolist())
            else:
                for _ in range(min(n_triplets//max(len(np.unique(chrom)),1),len(ind))):
                    triples.append(ind[np.sort(rng.choice(len(ind),3,replace=False))].tolist())
        if not triples: continue
        triples=np.asarray(triples,dtype=int)
        r,s=moments(real,triples),moments(syn,triples)
        key=f'gap_{step}'
        mask=np.isfinite(r)&np.isfinite(s)
        plot_xy(r[mask],s[mask],out/f'three_point_{key}.png',
                'Real third-order dosage moment','Synthetic third-order dosage moment',
                f'Genotype three-point correlation: {step}')
        records.append({'gap':str(step),'n_triplets':int(mask.sum()),'pearson_r':_corr(r,s),
                        'mae':float(np.mean(abs(r[mask]-s[mask]))) if mask.any() else np.nan})
    pd.DataFrame(records).to_csv(out/'three_point_summary.csv',index=False)
    return records


def motif_summary(real, syn, snps, out, max_windows=256, seed=42):
    chrom,bp,_=chromosome_positions(snps); rng=np.random.default_rng(seed)
    results={}
    for k in (4,8):
        windows=[]
        for ch in np.unique(chrom):
            ii=np.where(chrom==ch)[0];ii=ii[np.argsort(bp[ii],kind='stable')]
            windows.extend([ii[j:j+k] for j in range(0,len(ii)-k+1,k)])
        if len(windows)>max_windows:
            windows=[windows[i] for i in sorted(rng.choice(len(windows),max_windows,replace=False))]
        if not windows: continue
        w=np.array(windows)
        powers=(3**np.arange(k-1,-1,-1)).astype(np.int64)
        rr=[];ss=[]
        for indices in w:
            ar=(real[:,indices].astype(np.int64)*powers).sum(1)
            aS=(syn[:,indices].astype(np.int64)*powers).sum(1)
            codes=np.union1d(np.unique(ar),np.unique(aS))
            cr=np.bincount(ar,minlength=3**k)[codes]/len(real)
            cs=np.bincount(aS,minlength=3**k)[codes]/len(syn)
            rr.extend(cr);ss.extend(cs)
        rr=np.array(rr);ss=np.array(ss)
        plot_xy(rr,ss,out/f'genotype_{k}mer.png',
                'Real genotype-pattern frequency','Synthetic genotype-pattern frequency',
                f'{k}-SNP non-overlapping genotype motifs', (0,1))
        results[str(k)]={'windows':int(len(w)), 'observed_window_pattern_pairs':int(len(rr)),
                         'frequency_r':_corr(rr,ss),'frequency_mae':float(abs(rr-ss).mean())}
    return results


def _hamming(a,b):
    return pairwise_distances(a,b,metric='hamming',n_jobs=1)


def _balance(real,syn,cap,seed):
    rng=np.random.default_rng(seed)
    n=min(len(real),len(syn),cap)
    if n<2:raise ValueError('At least 2 samples needed in both datasets for distance metrics')
    return real[rng.choice(len(real),n,replace=False)],syn[rng.choice(len(syn),n,replace=False)]


def aats(real,syn,max_samples=256,seed=42,return_neighbors=False):
    """Paper's AATS definition with unphased genotype Hamming distance."""
    x,y=_balance(real,syn,max_samples,seed)
    xx=_hamming(x,x);yy=_hamming(y,y);xy=_hamming(x,y)
    np.fill_diagonal(xx,np.inf);np.fill_diagonal(yy,np.inf)
    truth=float(np.mean(np.min(xy,axis=1)>np.min(xx,axis=1)))
    fake=float(np.mean(np.min(xy,axis=0)>np.min(yy,axis=1)))
    result={'aa_truth':truth,'aa_synthetic':fake,'aats':(truth+fake)/2,
            'n_each':len(x),'hamming_unit':'fraction of diploid SNP genotypes unequal'}
    if return_neighbors: return result,(x,y,xx,yy,xy)
    return result


def distance_privacy(real,syn,plots,max_samples=256,seed=42,holdout=None):
    aa, (x,y,xx,yy,xy)=aats(real,syn,max_samples,seed,True)
    within_real=xx[np.triu_indices(len(x),1)]
    within_synthetic=yy[np.triu_indices(len(y),1)]
    between=xy.ravel()
    fig,ax=plt.subplots(figsize=(7,4))
    for a,label in [(within_real,'real-real'),(within_synthetic,'synthetic-synthetic'),(between,'real-synthetic')]:
        ax.hist(a,bins=35,alpha=.5,density=True,label=label)
    ax.legend(); ax.set(xlabel='Fraction of unequal diploid SNPs (Hamming distance)',ylabel='Density')
    fig.tight_layout();fig.savefig(plots/'pairwise_distances.png',dpi=155);plt.close(fig)
    aa['within_between_wasserstein']=float(wasserstein_distance(within_real,between))
    # Full-cohort exact replay using hashes; unlike NN statistics, not subsampled.
    real_rows={r.tobytes() for r in np.ascontiguousarray(real)}
    aa['exact_replay_full_count']=int(sum(row.tobytes() in real_rows for row in np.ascontiguousarray(syn)))
    # Allele-copy distance complements genotype-Hamming (0 vs 2 is twice 0 vs 1).
    manhattan_rr=pairwise_distances(x,metric='manhattan')[np.triu_indices(len(x),1)]/(2*x.shape[1])
    manhattan_ss=pairwise_distances(y,metric='manhattan')[np.triu_indices(len(y),1)]/(2*y.shape[1])
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(manhattan_rr,bins=30,alpha=.5,density=True,label='real-real')
    ax.hist(manhattan_ss,bins=30,alpha=.5,density=True,label='synthetic-synthetic')
    ax.legend();ax.set(xlabel='Mean absolute alternative-allele dosage difference / 2',ylabel='Density',title='Diploid allele-copy differences')
    fig.tight_layout();fig.savefig(plots/'pairwise_allele_copy_distances.png',dpi=155);plt.close(fig)
    aa['allele_copy_distance_wasserstein']=float(wasserstein_distance(manhattan_rr,manhattan_ss))
    aa['exact_real_synthetic_match_count_subsampled']=int(np.sum(np.min(xy,axis=0)==0))
    # Chains, start from each point and exclude already visited entries.
    full=np.block([[xx,xy],[xy.T,yy]])
    np.fill_diagonal(full,np.inf)
    counts={}
    for start in range(2*len(x)):
        used={start}; current=start; word='T' if start<len(x) else 'S'
        for _ in range(4):
            d=full[current].copy();d[list(used)]=np.inf
            if not np.isfinite(d).any():break
            current=int(np.argmin(d));used.add(current)
            word += 'T' if current<len(x) else 'S'
            counts[word]=counts.get(word,0)+1
    dataframe=pd.DataFrame([{'pattern':key,'length':len(key),'frequency':v/(2*len(x))}
                            for key,v in counts.items()])
    dataframe.to_csv(plots/'nearest_neighbor_chain_frequencies.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    top=dataframe[dataframe.length<=3].sort_values('pattern')
    ax.bar(top.pattern,top.frequency)
    ax.set(ylabel='Fraction of start points',title='Nearest-neighbor chain patterns (genotype Hamming)')
    fig.tight_layout();fig.savefig(plots/'nearest_neighbor_chains.png',dpi=155);plt.close(fig)
    if holdout is not None:
        test=aats(holdout,syn,max_samples,seed)  # matched subsampling seed for AATS difference
        aa['test_aats']=test['aats']
        aa['privacy_score_test_minus_train_aats']=test['aats']-aa['aats']
        aa['privacy_score_interpretation']='Only valid if real=actual TRAIN subjects and holdout=excluded TEST subjects; cohort membership must be proven.'
        fig,ax=plt.subplots(figsize=(5,4));ax.bar(['train','holdout'],[aa['aats'],test['aats']]);ax.axhline(.5,linestyle='--',color='black')
        ax.set(ylabel='AATS',title='Train-vs-holdout AATS')
        fig.tight_layout();fig.savefig(plots/'privacy_train_test_aats.png',dpi=155);plt.close(fig)
    return aa


def pca_comparison(real,syn,plots,seed=42,max_samples=2048):
    rng=np.random.default_rng(seed)
    if len(real)>max_samples: real=real[rng.choice(len(real),max_samples,replace=False)]
    if len(syn)>max_samples: syn=syn[rng.choice(len(syn),max_samples,replace=False)]
    n=min(4,len(real)+len(syn)-1,real.shape[1]);
    if n<2:return {}
    # Joint PCA projection, unlike reference-fit PCA in the training pipeline.
    a=np.vstack([real,syn]).astype(np.float32)
    pcs=PCA(n_components=n,random_state=seed).fit_transform(a)
    for i,j in [(0,1),(2,3)]:
        if n<=j:continue
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(pcs[:len(real),i],pcs[:len(real),j],alpha=.45,s=9,label='real')
        ax.scatter(pcs[len(real):,i],pcs[len(real):,j],alpha=.45,s=9,label='synthetic')
        ax.legend();ax.set(xlabel=f'PC{i+1}',ylabel=f'PC{j+1}',title='Combined real/synthetic PCA')
        fig.tight_layout();fig.savefig(plots/f'joint_pca_pc{i+1}_pc{j+1}.png',dpi=155);plt.close(fig)
        # Article-style per-cohort density contours on the same jointly fitted PC axes.
        if len(real)>=8 and len(syn)>=8:
            fig,axes=plt.subplots(1,2,figsize=(9,4.2),sharex=True,sharey=True)
            for ax,values,title in [(axes[0],pcs[:len(real)],'Real'),
                                    (axes[1],pcs[len(real):],'Synthetic')]:
                coords=values[:,[i,j]].T.astype(float)
                try:
                    kde=gaussian_kde(coords)
                    all_coords=pcs[:,[i,j]]
                    xx=np.linspace(np.min(all_coords[:,0]),np.max(all_coords[:,0]),65)
                    yy=np.linspace(np.min(all_coords[:,1]),np.max(all_coords[:,1]),65)
                    gx,gy=np.meshgrid(xx,yy)
                    dens=kde(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
                    if np.max(dens)>0 and np.ptp(dens)>0:
                        ax.contour(gx,gy,dens,levels=7)
                    else:ax.scatter(*coords,s=5,alpha=.3)
                except (np.linalg.LinAlgError,ValueError):
                    ax.scatter(*coords,s=5,alpha=.3)
                ax.set(title=title,xlabel=f'PC{i+1}',ylabel=f'PC{j+1}')
            fig.suptitle(f'Joint PCA: real and synthetic density (PC{i+1}/PC{j+1})')
            fig.tight_layout();fig.savefig(plots/f'joint_pca_density_pc{i+1}_pc{j+1}.png',dpi=155);plt.close(fig)
    return {'joint_pca_explained_variance':PCA(n_components=n,random_state=seed).fit(a).explained_variance_ratio_.tolist()}


def radar_summary(metrics,plots,real,syn,seed):
    # These are bounded diagnostic transformations, NOT the exact paper's
    # binomial-model-baseline min/max scaling unless that baseline is provided.
    ld=metrics['ld'].get('all_chromosomes',{})
    af=metrics['frequencies']; aa=metrics['privacy']
    trip=[r['pearson_r'] for r in metrics['three_point'] if r['gap']=='random']
    motifs=metrics['motifs']
    names=['Rare AF','LD decay/pairs','Three-point','4-mer','8-mer','Overfit','Underfit','Pairwise distance']
    aa_t=aa.get('aa_truth',float('nan'));aa_s=aa.get('aa_synthetic',float('nan'))
    # Match the 2023 paper's binomial null calibration for pairwise distances
    # to a genotype-only HWE null (not a claim that HWE holds in the cohort).
    rng=np.random.default_rng(seed+111)
    null=rng.binomial(2,real.mean(0)/2,size=syn.shape).astype(np.int8)
    real_s,syn_s=_balance(real,syn,256,seed+112)
    real_n,null_s=_balance(real,null,256,seed+112)
    rr=_hamming(real_s,real_s)[np.triu_indices(len(real_s),1)]
    ss=_hamming(syn_s,syn_s)[np.triu_indices(len(syn_s),1)]
    nn=_hamming(null_s,null_s)[np.triu_indices(len(null_s),1)]
    w_syn=wasserstein_distance(rr,ss)
    w_null=wasserstein_distance(rr,nn)
    dist_score=float(np.clip(1-w_syn/w_null,0,1)) if w_null>1e-12 else float('nan')
    vals=[af['af_low_frequency_r'],ld.get('ld_decay_radar_score',float('nan')),
          trip[0] if trip else float('nan'),motifs.get('4',{}).get('frequency_r',float('nan')),
          motifs.get('8',{}).get('frequency_r',float('nan')),
          1-max(0,.5-aa_t)-max(0,.5-aa_s),
          1-max(0,aa_t-.5)-max(0,aa_s-.5),
          dist_score]
    vals=np.array(vals,float)
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='polar')
    th=np.linspace(0,2*np.pi,len(vals),endpoint=False)
    y=np.nan_to_num(np.clip(vals,0,1),nan=0.)
    ax.plot(np.r_[th,th[0]],np.r_[y,y[0]],'o-')
    ax.fill(np.r_[th,th[0]],np.r_[y,y[0]],alpha=.15)
    ax.set_xticks(th,names);ax.set_ylim(0,1);ax.set_title('Adapted fidelity radar (not original paper scaling)')
    fig.tight_layout();fig.savefig(plots/'adapted_radar.png',dpi=150);plt.close(fig)
    return dict(zip(names,[None if not np.isfinite(x) else float(x) for x in vals]))


def evaluate_paper_analyses(real, synthetic, snps, output_dir, seed=42,
                            max_ld_pairs=3000,max_distance_samples=256,
                            max_motif_windows=256,max_triplets=2000,
                            holdout=None):
    """Produce figure families once, for any number of chromosomes/SNPs."""
    real=_clean_geno(real,'real');synthetic=_clean_geno(synthetic,'synthetic')
    snps=np.asarray(snps,dtype=str)
    if real.shape[1]!=synthetic.shape[1] or real.shape[1]!=len(snps):
        raise ValueError('Real and synthetic matrices must have identical ordered SNP columns')
    if holdout is not None: holdout=_clean_geno(holdout,'holdout')
    out=Path(output_dir);plots=out/'plots';metrics_dir=out/'metrics'
    plots.mkdir(parents=True,exist_ok=True);metrics_dir.mkdir(parents=True,exist_ok=True)
    summary={'n_real':int(len(real)),'n_synthetic':int(len(synthetic)),'n_snps':int(len(snps)),
             'representation':'diploid unphased 0/1/2, no phase reconstructed',
             'frequencies':frequency_statistics(real,synthetic,plots),
             'diploid':genotype_frequency_stats(real,synthetic,plots)}
    summary['pca']=pca_comparison(real,synthetic,plots,seed)
    summary['ld'],fallback=ld_summary(real,synthetic,snps,plots/'ld',max_ld_pairs,seed)
    summary['ld_position_fallback_present']=fallback
    summary['ld_definition']='Dosage Pearson r squared, within chromosome only; pooled curve pools within-chromosome pairs'
    summary['phased_haplotype_metrics_not_identifiable']=['phase-specific haplotype motifs','haplotype sharing/painting','phased k-point moments']
    summary['three_point']=triplet_summary(real,synthetic,snps,plots,max_triplets,seed)
    summary['motifs']=motif_summary(real,synthetic,snps,plots,max_motif_windows,seed)
    summary['privacy']=distance_privacy(real,synthetic,plots,max_distance_samples,seed,holdout)
    summary['radar']=radar_summary(summary,plots,real,synthetic,seed)
    with open(metrics_dir/'paper_analyses.json','w') as fh:json.dump(summary,fh,indent=2,default=_json)
    pd.json_normalize(summary,sep='.').to_csv(metrics_dir/'paper_analyses_flat.csv',index=False)
    return summary
