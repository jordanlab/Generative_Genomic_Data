"""LOGAN-style white-box critic ranking and optional black-box surrogate critic.

Requires *proven* training-vs-excluded subjects and correct conditions. The
black-box option retrains a PCA-space conditional WGAN on generated genotypes
and is labelled an approximation when the original architecture differs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def critic_scores(model, backend, geno, conditions, pca_components, pca_mean,
                  batch=128):
    if len(geno) != len(conditions): raise ValueError('One ancestry condition per scored sample required')
    x=np.asarray(geno,dtype=np.float32)
    if backend!='genotype_conv':
        pc=(x-pca_mean) @ pca_components.T
        norm=(pc-model._mean)/model._std
    else: norm=x-1
    vals=[]
    model._C.eval()
    pack=1 if backend=='baseline_pca' else model.config.pack_size
    with torch.no_grad():
        for i in range(0,len(geno),batch):
            xi=torch.from_numpy(norm[i:i+batch]).to(model._device)
            ci=torch.from_numpy(np.asarray(conditions[i:i+batch],dtype=np.float32)).to(model._device)
            if pack>1:
                # Duplicate each sample through the full pack: an individual
                # diagnostic proxy, not the original joint 3-sample statistic.
                if backend=='genotype_conv':
                    xi=xi.unsqueeze(1).repeat(1,pack,1)
                else: xi=xi.repeat(1,pack)
                ci=ci.repeat(1,pack)
            val=model._C(xi,ci).reshape(-1).cpu().numpy()
            vals.extend(val.tolist())
    return np.asarray(vals,dtype=float)


def attack_from_scores(train_scores,test_scores,plot_dir,name):
    plot_dir=Path(plot_dir);plot_dir.mkdir(parents=True,exist_ok=True)
    scores=np.r_[train_scores,test_scores]
    labels=np.r_[np.ones(len(train_scores),dtype=int),np.zeros(len(test_scores),dtype=int)]
    order=np.argsort(-scores)
    rates=[]
    for f in (.01,.10,.25,.50):
        n=max(1,int(np.ceil(len(scores)*f)))
        precision=float(np.mean(labels[order[:n]]))
        rates.append({'top_fraction':f,'n_selected':n,
                      'member_precision_at_n':precision,
                      'random_baseline_member_fraction':float(labels.mean())})
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r['top_fraction']*100 for r in rates],[r['member_precision_at_n'] for r in rates],'o-')
    ax.axhline(labels.mean(),linestyle='--',color='red',label='random baseline')
    ax.set(xlabel='Highest-ranked fraction (%)',ylabel='Member precision among selected',
           title=f'Critic-score membership inference: {name}')
    ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(plot_dir/f'mia_{name}_ranking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(train_scores,bins=25,alpha=.5,label='actual training')
    ax.hist(test_scores,bins=25,alpha=.5,label='excluded test')
    ax.set(xlabel='Critic score',ylabel='Count',title=f'Critic scores: {name}')
    ax.legend();fig.tight_layout();fig.savefig(plot_dir/f'mia_{name}_score_distribution.png',dpi=160);plt.close(fig)
    pd.DataFrame(rates).to_csv(plot_dir/f'mia_{name}_thresholds.csv',index=False)
    return rates


def whitebox_attack(model,backend,train,holdout,c_train,c_test,
                    pca_components,pca_mean,output_dir):
    if len(train)<2 or len(holdout)<2:
        return {'status':'skipped','reason':'Insufficient held-out and train subjects'}
    a=critic_scores(model,backend,train,c_train,pca_components,pca_mean)
    b=critic_scores(model,backend,holdout,c_test,pca_components,pca_mean)
    return {'status':'computed','score_definition':'original conditional critic; larger rank predicts member',
            'packed_critic_note':'sample repeated within its own pack' if getattr(getattr(model,'config',None),'pack_size',1)>1 else None,
            'ranking':attack_from_scores(a,b,output_dir,'whitebox')}


def blackbox_surrogate_attack(train,holdout,c_train,c_test,
                              synthetic,c_synthetic,components,pca_mean,
                              output_dir,epochs=0,seed=42,device='cpu'):
    if epochs<=0:return {'status':'not_run','reason':'set mia_blackbox_epochs>0 to train a surrogate GAN critic'}
    # Auxiliary WGAN trained ONLY on synthetic samples. Baseline MLP critic
    # deliberately shared across the three target backends for comparability.
    # It is not a reproduction of a convolutional target's architecture.
    from conditional_wgan_pipeline_extended import ConditionalWGAN
    pc_syn=(np.asarray(synthetic,dtype=np.float32)-pca_mean)@components.T
    proxy=ConditionalWGAN(latent_dim=32,cond_dim=3,hidden_dims=[64,64],
                          n_epochs=epochs,batch_size=min(32,max(2,len(synthetic)//2)),
                          n_critic=1,val_fraction=0,log_every=0,seed=seed,device=device)
    proxy.fit(pc_syn,np.asarray(c_synthetic,dtype=np.float32))
    a=critic_scores(proxy,'baseline_pca',train,c_train,components,pca_mean)
    b=critic_scores(proxy,'baseline_pca',holdout,c_test,components,pca_mean)
    return {'status':'computed','surrogate_architecture':'PCA-space MLP conditional WGAN, trained only on synthetic samples',
            'surrogate_epochs':epochs,'not_exact_paper_architecture':True,
            'ranking':attack_from_scores(a,b,output_dir,'blackbox_surrogate')}
