"""Retrospective evaluation and optional resampling of archived WGAN runs.

Examples:
    python evaluate_saved_genotypes.py --evaluate_run runs/20261004_105931_sample_size_sweep
    python evaluate_saved_genotypes.py --evaluate_run runs/... --resample \
        --sampling_methods binomial categorical ternary ternary_normal --weights best last
    python evaluate_saved_genotypes.py --evaluate_run runs/... --real_input BOG=data/BOG.csv
    # Only BOG_* jobs are selected by the explicit BOG mapping.
    # Without any --real_input, paths saved in each model's args.json are used.
"""
import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from genotype_sampling import METHODS, sample_dosages, inspect_raw, plot_pca_sampling_diagnostics
from genotype_paper_evaluation import evaluate_paper_analyses
from genotype_input_qc import audit as audit_genotypes, write_bundle as write_genotype_qc
from cwgan_extensions.models import FiLMConditionalWGAN, GenotypeConvWGAN
from conditional_wgan_pipeline_extended import ConditionalWGAN, _CondNetG, _CondNetC


def _torch_load_trusted(path):
    # Never load checkpoint pickle files obtained from untrusted origins.
    return torch.load(path, map_location='cpu', weights_only=False)


def _device(value):
    return torch.device('cuda' if (value in (None,'auto') and torch.cuda.is_available()) else
                        'cpu' if value in (None,'auto') else value)


def load_checkpoint_model(path, pca_path, device='auto'):
    ck=_torch_load_trusted(path)
    with np.load(pca_path,allow_pickle=False) as data:
        components=data['components'].astype(np.float32)
        pca_mean=data['mean'].astype(np.float32)
        snps=data['snp_names'].astype(str)
        stored_sign_flag='pc_sign_applied_to_saved_model' in data.files
    n_pc=components.shape[0];n_snps=len(snps)
    if not stored_sign_flag:
        import warnings
        warnings.warn('Legacy saved PCA lacks post-orientation marker; PC1 could have been flipped after saving. Verify resampled predictions against original output.',RuntimeWarning)
    dev=_device(device)
    if 'hyperparameters' in ck:  # Original baseline PCA model format
        h=ck['hyperparameters']
        model=ConditionalWGAN(**{k:v for k,v in h.items()
                                 if k in inspect.signature(ConditionalWGAN).parameters},device=str(dev))
        model._G=_CondNetG(model.latent_dim,model.cond_dim,model.hidden_dims,n_pc,
                            model.leaky_relu_slope,model.layernorm_eps).to(dev)
        model._C=_CondNetC(n_pc,model.cond_dim,model.hidden_dims,model.leaky_relu_slope).to(dev)
        model._mean=np.asarray(ck['pc_mean'],dtype=np.float32)
        model._std=np.asarray(ck['pc_std'],dtype=np.float32)
        backend='baseline_pca'
    else:
        kind=ck.get('model_kind')
        config=ck['config'];extra=ck.get('extra_state',{})
        if kind=='pca_film_mlp':
            options=dict(config,hidden_dims=extra.get('hidden_dims'),
                         leaky_relu_slope=extra.get('leaky_relu_slope',.2),
                         layernorm_eps=extra.get('layernorm_eps',1e-5))
            klass=FiLMConditionalWGAN; size=n_pc;backend='pca_film'
        elif kind=='genotype_conv1d':
            options=dict(config,n_snps=extra.get('n_snps',n_snps),
                         channels=extra.get('channels',128),n_blocks=extra.get('n_blocks',4),
                         position_channel=extra.get('position_channel',True),
                         position_bias=extra.get('position_bias',True))
            klass=GenotypeConvWGAN; size=n_snps;backend='genotype_conv'
        else: raise ValueError(f'Unsupported checkpoint format/model_kind: {kind}')
        options={k:v for k,v in options.items() if k in inspect.signature(klass).parameters}
        options['device']=str(dev)
        model=klass(**options)
        model._G,model._C=model._build_networks(size)
        model._G=model._G.to(dev);model._C=model._C.to(dev)
        if backend=='pca_film':
            model._mean=np.asarray(extra['mean'],dtype=np.float32)
            model._std=np.asarray(extra['std'],dtype=np.float32)
    model._G.load_state_dict(ck['generator_state_dict'])
    model._C.load_state_dict(ck['critic_state_dict'])
    model._G.eval();model._C.eval();model._device=dev
    return model,backend,snps,pca_mean,components,ck


def _is_model_dir(path):
    return (path / 'args.json').is_file() and (path / 'checkpoints').is_dir()


def _model_dirs(run):
    if _is_model_dir(run):
        return [run]
    if not run.is_dir():
        raise NotADirectoryError(f'Not a run/model directory: {run}')
    return sorted(x for x in run.iterdir() if x.is_dir() and _is_model_dir(x))


_SINGLE_SUBSET = re.compile(r'^(?P<member>[A-Za-z][A-Za-z0-9]*)_(?P<subset>all|n[0-9]+)$', re.I)


def _model_members(model_dir, args):
    """Resolve authoritative cohort names, and cross-check simple sweep folders.

    The per-model args.json written during training has `dataset_members`.
    In a legacy archive, `dataset_paths` or a simple folder name is the fallback.
    A folder named BOG_n25 must *never* be treated as another cohort solely
    because stale/corrupt args.json claims it is one.
    """
    folder = _SINGLE_SUBSET.fullmatch(model_dir.name)
    declared = args.get('dataset_members')
    if not declared:
        declared = list((args.get('dataset_paths') or {}).keys())
    if not declared and folder:
        declared = [folder.group('member')]
    if not isinstance(declared, (list, tuple)) or not declared:
        raise ValueError(f'{model_dir}: cannot determine dataset members from args.json; '
                         'expected dataset_members or dataset_paths')
    members = tuple(str(x).strip().upper() for x in declared)
    if len(members) != len(set(members)) or any(not m for m in members):
        raise ValueError(f'{model_dir}: invalid/duplicate dataset_members: {declared}')
    if folder and members != (folder.group('member').upper(),):
        raise ValueError(f'{model_dir}: folder identifies {folder.group("member")}, '
                         f'but args.json identifies {members}. Refusing to mix cohorts.')
    return members


def _select_model_dirs(run, overrides):
    """An explicit NAME=PATH list selects only matching training jobs.

    Combined jobs are selected only if *all* their cohort names were explicitly
    provided, never by silently filling in another cohort from stored args.
    With no overrides, evaluate every model with its own stored dataset paths.
    """
    candidates = _model_dirs(run)
    if not candidates:
        raise ValueError(f'No model directories containing args.json and checkpoints found under {run}')
    selected, excluded = [], []
    for model_dir in candidates:
        args = json.loads((model_dir / 'args.json').read_text())
        members = _model_members(model_dir, args)
        missing = sorted(set(members) - set(overrides or {}))
        if overrides and missing:
            excluded.append({'model_dir': str(model_dir), 'members': list(members),
                             'reason': f'explicit --real_input excludes {", ".join(missing)}'})
            continue
        selected.append((model_dir, args, members))
    if not selected:
        raise ValueError(f'No model directories matched --real_input names '
                         f'{sorted(overrides or {})}; available: '
                         f'{[(x.name, _model_members(x, json.loads((x / "args.json").read_text()))) for x in candidates]}')
    return selected, excluded


def _real_from_model_args(model_dir, args, snps, real_override=None, members=None):
    members = tuple(members or _model_members(model_dir, args))
    stored = {str(k).upper(): v for k, v in (args.get('dataset_paths') or {}).items()}
    if real_override:
        # Never substitute another dataset's CSV from args.json when an explicit
        # cohort filter was supplied (including for combined training jobs).
        paths = {m: real_override[m] for m in members}
    else:
        paths = {m: stored.get(m) for m in members}
    missing = [m for m in members if not paths.get(m)]
    if missing:
        raise ValueError(f'{model_dir}: real input paths missing for {missing}; '
                         'pass --real_input NAME=PATH for each member')

    subset_path = model_dir / 'outputs' / 'dataset_subset_indices.json'
    subsets = json.loads(subset_path.read_text()) if subset_path.exists() else {}
    choices = {str(k).upper(): v for k, v in
               (args.get('subset_choices_for_model') or {}).items()}
    folder = _SINGLE_SUBSET.fullmatch(model_dir.name)
    tables = []
    for member in members:
        path = Path(paths[member]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{model_dir}: missing {member} real genotypes: {path}; '
                                    f'pass --real_input {member}=PATH')
        df = pd.read_csv(path)
        missing_snps = [s for s in snps if s not in df.columns]
        if missing_snps:
            raise ValueError(f'{model_dir}: {member} file {path} lacks '
                             f'{len(missing_snps)} model SNPs, e.g. {missing_snps[:4]}')
        record = subsets.get(member)
        selected = record.get('selected_row_indices_zero_based') if isinstance(record, dict) else None
        need_subset = (isinstance(choices.get(member), int) or
                       (folder is not None and folder.group('subset').lower().startswith('n')))
        if selected is None and need_subset:
            raise ValueError(f'{model_dir}: missing recorded training subset indices for {member}. '
                             f'Expected {subset_path}; refusing to compare an n-subset '
                             'generator to the full original dataset.')
        if selected is not None:
            selected = np.asarray(selected)
            if selected.ndim != 1 or selected.dtype.kind not in 'iu':
                raise ValueError(f'{subset_path}: {member} has non-integer indices')
            if (selected.size == 0 or np.unique(selected).size != selected.size or
                    np.any(selected < 0) or np.any(selected >= len(df))):
                raise ValueError(f'{subset_path}: {member} indices do not fit {path}; '
                                 'the supplied CSV must be the original, same-order source')
            original_n = record.get('original_n_samples')
            if original_n is not None and int(original_n) != len(df):
                raise ValueError(f'{model_dir}: {member} source {path} has {len(df)} rows, '
                                 f'but original run had {original_n}; row indices might select '
                                 'different individuals. Supply the original same-order CSV.')
            expected_n = int(record.get('used_n_samples', len(selected)))
            if len(selected) != expected_n:
                raise ValueError(f'{subset_path}: {member} subset length {len(selected)} '
                                 f'differs from recorded {expected_n}')
            if folder and len(members) == 1 and folder.group('subset').lower().startswith('n'):
                folder_n = int(folder.group('subset')[1:])
                if len(selected) != folder_n:
                    raise ValueError(f'{model_dir}: recorded subset length {len(selected)} '
                                     f'disagrees with folder label n{folder_n}')
            if isinstance(choices.get(member), int) and len(selected) != choices[member]:
                raise ValueError(f'{model_dir}: {member} subset size contradicts args.json')
            df = df.iloc[selected]
        tables.append(df.loc[:, list(snps)])
    return pd.concat(tables, ignore_index=True)


def _conditioning_csv_for(syn_path):
    """Return corresponding saved conditions, handling resampling artifacts.

    Historical resampling only wrote *_resampled.csv, so those old outputs
    can be evaluated if it still matches in shape. New resampling stores a
    seed-specific condition file to avoid reusing conditions from a later run.
    """
    outputs = syn_path.parent
    m = re.search(r'_resample_seed_(\d+)$', syn_path.stem)
    if m:
        same_seed = outputs / f'synthetic_conditioning_admixture_resample_seed_{m.group(1)}.csv'
        if same_seed.is_file():
            return same_seed
        return outputs / 'synthetic_conditioning_admixture_resampled.csv'
    return outputs / 'synthetic_conditioning_admixture.csv'


def _validate_saved_pair(syn_path, n_samples):
    path = _conditioning_csv_for(syn_path)
    if not path.is_file():
        raise FileNotFoundError(f'{syn_path}: matching saved conditioning file missing: {path}')
    cond = pd.read_csv(path)
    required = ('AFR', 'EUR', 'AMR')
    if list(cond.columns) != list(required):
        raise ValueError(f'{path}: expected conditioning columns AFR,EUR,AMR in that order')
    if len(cond) != n_samples:
        raise ValueError(f'{syn_path}: {n_samples} genotypes but {path} has '
                         f'{len(cond)} conditioning rows; refuse mismatched saved pair')
    return path


def _discrete_comparison(real_df,syn_df,holdout_df=None):
    """Impute genuine missing markers from real sample locus modal genotype.

    Keeps the SNP set fixed; counts imputed entries and explicitly logs them.
    """
    snps=list(syn_df.columns)
    if not all(s in real_df.columns for s in snps):
        raise ValueError('Real data lack expected SNP columns')
    a=real_df[snps].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    b=syn_df[snps].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    c=holdout_df[snps].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float) if holdout_df is not None else None
    fill=np.zeros(len(snps),dtype=float)
    for j in range(len(snps)):
        observed=a[np.isfinite(a[:,j]) & np.isin(a[:,j],[0,1,2]),j]
        fill[j]=np.bincount(observed.astype(int),minlength=3).argmax() if len(observed) else 0
    imp={}
    for label,array in [('real',a),('synthetic',b),('holdout',c)]:
        if array is None:continue
        bad=~np.isin(array,[0,1,2])
        # Reject clearly invalid synthetic samples rather than hiding an error.
        if label=='synthetic' and bad.any():raise ValueError('Synthetic genotype outputs must be discrete 0,1,2')
        imp[label]=int(bad.sum())
        row,col=np.where(bad);array[row,col]=fill[col]
    return a.astype(np.int8),b.astype(np.int8),None if c is None else c.astype(np.int8),imp


def _load_saved_pca(model_dir):
    p=model_dir/'checkpoints'/'pca_model.npz'
    if not p.exists():raise FileNotFoundError(f'{p} required for resampling PCA-space model')
    with np.load(p,allow_pickle=False) as data:
        return data['snp_names'].astype(str)


def _checkpoint_for_labels(model_dir, labels):
    for label in labels:
        p=model_dir/'checkpoints'/f'checkpoint_{label}.pt'
        if p.exists():return p,label
    p=model_dir/'checkpoints'/'conditional_wgan_checkpoint.pt'
    if p.exists():return p,'compatibility'
    raise FileNotFoundError(f'No matching checkpoint in {model_dir}')


def _variant_labels(variant):
    stem=variant.name
    return [s for s in ('baseline','first','best','last') if s in stem.split('_')]


def _sampling_continuous(model,backend,cond,pc_mean,pcs_components,seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    cont=model.sample(len(cond),cond)
    return cont if backend=='genotype_conv' else cont @ pcs_components + pc_mean


def _save_summary(summaries, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(summaries,indent=2,allow_nan=True))


def evaluate_run(run,real_overrides=None,holdout_path=None,resample=False,
                 methods=('binomial',),weights=('best','last','baseline','first'),
                 seed=42,n_synthetic=None,device='auto',temperature=.35,
                 max_ld_pairs=3000,max_distance_samples=256,
                 max_motif_windows=256,max_triplets=2000,
                 experiment_name=None,plots_only=False):
    run = Path(run).expanduser().resolve()
    real_overrides = {str(k).upper(): v for k, v in (real_overrides or {}).items()}
    if n_synthetic is not None and n_synthetic < 1:
        raise ValueError('--n_synthetic must be a positive integer')
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('--sampling_temperature must be finite and > 0')
    if plots_only and not resample:
        raise ValueError('--plots_only requires --resample')
    if experiment_name is not None and not resample:
        raise ValueError('--experiment_name requires --resample')
    if not methods or any(x not in METHODS for x in methods):
        raise ValueError(f'Expected sampling methods from {METHODS}')
    if experiment_name is None:
        # Unique between sampler/temperature/seed configurations; stable, readable.
        import hashlib
        fingerprint = hashlib.sha256(json.dumps([list(methods), temperature, seed,
                             n_synthetic], sort_keys=True).encode()).hexdigest()[:10]
        experiment_name = f'sampling_seed_{seed}_{fingerprint}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', experiment_name) or experiment_name in ('.', '..'):
        raise ValueError('--experiment_name must be a safe, simple folder name (letters, numbers, _, -, .)')
    selected, excluded = _select_model_dirs(run, real_overrides)
    for skipped in excluded:
        print(f'[SKIP] {Path(skipped["model_dir"]).name}: {skipped["reason"]}', flush=True)
    all_results = {}
    omitted_outputs = []
    for model_dir, args, members in selected:
        print(f'[MODEL] {model_dir.name}: real dataset(s)={list(members)}, '
              f'input source={"--real_input" if real_overrides else "args.json"}', flush=True)
        saved_root = model_dir / 'saved_model_outputs'
        variants = sorted(p for p in saved_root.iterdir() if p.is_dir()) if saved_root.exists() else []
        results = []
        real_cache = {}
        # Default: look ONLY for already sampled CSV pairs. Checkpoints are not
        # opened and no sampling methods are invoked unless --resample is set.
        if resample:
            targets = []
            used = set()
            for label in weights:
                ckpt = model_dir / 'checkpoints' / f'checkpoint_{label}.pt'
                if not ckpt.is_file():
                    print(f'[SKIP] {model_dir.name}: no checkpoint_{label}.pt', flush=True)
                    continue
                ck = _torch_load_trusted(ckpt)
                epoch = (ck.get('best_epoch') if label == 'best' else
                         ck.get('last_epoch') if label == 'last' else
                         0 if label == 'baseline' else 1)
                state = str(epoch)  # checkpoint labels at the same epoch share one output
                if state in used:
                    continue
                used.add(state)
                matching = next((v for v in variants if label in _variant_labels(v)), None)
                if matching is None:
                    matching = saved_root / f'epoch_{int(epoch or 0):05d}_{label}'
                targets.append((matching, ckpt, label))
        else:
            targets = [(v, None, 'existing') for v in variants]

        for variant, ckpt_path, label in targets:
            original_outputs = variant / 'outputs'
            # Resampling NEVER writes into the original checkpoint's outputs,
            # plots, diagnostics, or paper_evaluation directories.
            experiment_dir = variant / 'sampling_experiments' / experiment_name if resample else variant
            if resample:
                if experiment_dir.exists():
                    raise FileExistsError(f'{experiment_dir}: experiment already exists. '
                                          'Choose a different --experiment_name to avoid overwriting results.')
                experiment_dir.mkdir(parents=True)
            outputs = experiment_dir / 'outputs'
            primary = outputs / 'synthetic_genotypes.csv'
            if resample:
                model, backend, snps, pc_mean, pcs_components, ck = load_checkpoint_model(
                    ckpt_path, model_dir / 'checkpoints' / 'pca_model.npz', device)
                conditions_path = original_outputs / 'synthetic_conditioning_admixture.csv'
                if not conditions_path.is_file():
                    conditions_path = next((v / 'outputs' / 'synthetic_conditioning_admixture.csv'
                                            for v in variants if
                                            (v / 'outputs' / 'synthetic_conditioning_admixture.csv').is_file()), None)
                if conditions_path is None or not conditions_path.is_file():
                    raise FileNotFoundError(f'{model_dir}: resampling requires saved ancestry '
                                            'conditions from an existing checkpoint output')
                conditions = pd.read_csv(conditions_path)
                if list(conditions.columns) != ['AFR', 'EUR', 'AMR']:
                    raise ValueError(f'{conditions_path}: expected AFR,EUR,AMR columns')
                cond = conditions.to_numpy(dtype=np.float32)
                size = n_synthetic if n_synthetic is not None else len(cond)
                if len(cond) != size:
                    cond = cond[np.random.default_rng(seed).choice(len(cond), size=size, replace=True)]
                outputs.mkdir(parents=True, exist_ok=True)
                # The conditions, data, preclip diagnostics, PCA comparisons
                # and per-method samples live together in the experiment folder.
                cond_file = outputs / 'synthetic_conditioning_admixture.csv'
                pd.DataFrame(cond, columns=['AFR', 'EUR', 'AMR']).to_csv(cond_file, index=False)
                raw = _sampling_continuous(model, backend, cond, pc_mean, pcs_components, seed)
                inspect_raw(raw, experiment_dir / 'sampling_diagnostics', seed, snp_names=snps)
                # Use the actual selected training subset as the real comparison set;
                # missing sentinel calls are filled with training-subset SNP means.
                real_df_for_plot = _real_from_model_args(
                    model_dir, args, snps, real_overrides or None, members)
                real_values = real_df_for_plot.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
                bad = ~np.isfinite(real_values) | (real_values < 0)
                valid = np.where(bad, np.nan, real_values)
                count = np.sum(np.isfinite(valid), axis=0)
                means = np.divide(np.nansum(valid, axis=0), count,
                                  out=np.zeros(raw.shape[1], dtype=np.float64), where=count > 0)
                row, col = np.nonzero(bad)
                real_values[row, col] = means[col]
                def project(x):
                    return (np.asarray(x, dtype=np.float64) - pc_mean) @ pcs_components.T
                real_pcs = project(real_values)
                continuous_pcs = project(raw)
                samples = {method: sample_dosages(raw, method, seed, temperature)
                           for method in methods}
                sampled_pcs = {method: project(sample) for method, sample in samples.items()}
                plot_pca_sampling_diagnostics(
                    real_pcs, continuous_pcs, sampled_pcs, experiment_dir / 'plots',
                    label=f'Synthetic {backend}',
                )
                candidates = []
                for method, sample in samples.items():
                    dest = outputs / f'synthetic_genotypes_{method}.csv'
                    pd.DataFrame(sample, columns=snps).to_csv(dest, index=False)
                    candidates.append((method, dest))
                _save_summary({
                    'mode': 'resample_and_plot', 'source_variant': str(variant),
                    'source_checkpoint': str(ckpt_path), 'model_backend': backend,
                    'sampling_methods': list(methods), 'sampling_temperature': temperature,
                    'seed': seed, 'n_synthetic': len(cond),
                    'real_genotypes_used_for_pca': str(real_df_for_plot.shape),
                    'real_missing_entries_mean_imputed_for_pca': int(bad.sum()),
                    'real_all_missing_snps_zero_fallback': int((count == 0).sum()),
                    'note': 'Existing best checkpoint is not re-selected; these are post-hoc sampler experiments. '
                            'Real PCA is projected with the original saved reference PCA. '
                            'Mean imputation uses the selected real subset, zero if fully missing.',
                }, experiment_dir / 'experiment_manifest.json')
            else:
                if not primary.is_file():
                    msg = (f'{variant}: missing saved outputs/synthetic_genotypes.csv; '
                           'use --resample to generate it')
                    print(f'[SKIP] {msg}', flush=True)
                    omitted_outputs.append(msg)
                    continue
                if not (outputs / 'synthetic_conditioning_admixture.csv').is_file():
                    msg = f'{variant}: missing saved outputs/synthetic_conditioning_admixture.csv; '
                    msg += 'will not treat this as a complete saved sample pair'
                    print(f'[SKIP] {msg}', flush=True)
                    omitted_outputs.append(msg)
                    continue
                candidates = [('existing', primary)]
                # Already-created alternative samples are evaluated, not regenerated.
                candidates.extend((p.stem.replace('synthetic_genotypes_', ''), p)
                                  for p in sorted(outputs.glob('synthetic_genotypes_*.csv')))

            for method, synthetic_file in candidates:
                if resample and plots_only:
                    # Fast diagnostics: all CSVs and matched PCA plots already
                    # exist. Do not run the costly full paper evaluation.
                    results.append({
                        'variant': variant.name, 'method': method,
                        'output': str(experiment_dir / 'plots'),
                        'experiment_dir': str(experiment_dir),
                        'source_synthetic': str(synthetic_file),
                        'source_conditioning': str(cond_file),
                        'n_real': len(real_df_for_plot), 'n_synthetic': len(cond),
                        'plots_only': True,
                    })
                    continue
                df_syn = pd.read_csv(synthetic_file)
                try:
                    cond_csv = _validate_saved_pair(synthetic_file, len(df_syn))
                except (FileNotFoundError, ValueError) as exc:
                    msg = f'{model_dir.name}/{variant.name}/{method}: {exc}'
                    if method == 'existing' or resample:
                        raise ValueError(msg) from exc
                    print(f'[SKIP] {msg}', flush=True)
                    omitted_outputs.append(msg)
                    continue
                columns = tuple(df_syn.columns)
                if columns not in real_cache:
                    real_cache[columns] = _real_from_model_args(
                        model_dir, args, columns, real_overrides or None, members)
                df_real = real_cache[columns]
                df_hold = pd.read_csv(holdout_path) if holdout_path else None
                real, syn, hold, imputed = _discrete_comparison(df_real, df_syn, df_hold)
                target = experiment_dir / 'paper_evaluation' / method
                raw_real=df_real.loc[:,list(columns)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
                raw_synth=df_syn.loc[:,list(columns)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
                qc_stages=[
                    audit_genotypes(raw_real,dataset='+'.join(members),stage='evaluation_real_before_discrete_imputation',missing_rule='nan'),
                    audit_genotypes(real,dataset='+'.join(members),stage='evaluation_real_after_modal_imputation',missing_rule='nan'),
                    audit_genotypes(raw_synth,dataset='synthetic',stage='saved_synthetic',missing_rule='nan')]
                # Evaluation deliberately retains its historical integer-only
                # modal conversion, unlike the mean-imputed continuous training
                # matrix. This difference must be visible in the report.
                write_genotype_qc(target/'input_qc',qc_stages,
                    per_snp={'evaluation_real_before_imputation':(raw_real,list(columns))} if method=='existing' else None,
                    metadata={'evaluation_discretization':'modal real-SNP hard calls; do not confuse with cWGAN mean-imputed training',
                              'input_imputations':imputed,'source_synthetic':str(synthetic_file)})
                print(f'[{model_dir.name} ({"+".join(members)})] {variant.name} / {method}: '
                      f'{"plotting" if plots_only else "evaluating"} {len(real)} real '
                      f'and {len(syn)} saved synthetic samples x {len(columns)} SNPs', flush=True)
                metrics = evaluate_paper_analyses(
                    real, syn, columns, target, seed,
                    max_ld_pairs, max_distance_samples, max_motif_windows, max_triplets,
                    holdout=hold)
                metrics['input_imputations_by_cohort'] = imputed
                metrics['source_syn_csv'] = str(synthetic_file)
                metrics['source_conditioning_csv'] = str(cond_csv)
                metrics['original_vs_resampled'] = method
                metrics['dataset_members'] = list(members)
                metrics['source_real_csvs'] = {
                    m: str(real_overrides[m] if real_overrides else
                           args.get('dataset_paths', {}).get(m, '')) for m in members}
                _save_summary(metrics, target / 'metrics' / 'paper_analyses.json')
                results.append({'variant': variant.name, 'method': method,
                                'output': str(target), 'imputations': imputed,
                                'dataset_members': list(members),
                                'source_synthetic': str(synthetic_file),
                                'source_conditioning': str(cond_csv),
                                'n_real': len(real), 'n_synthetic': len(syn),
                                'experiment_dir': str(experiment_dir)})
        if not targets:
            print(f'[SKIP] no matching saved checkpoints/outputs in {model_dir}', flush=True)
        _save_summary(results, (model_dir / 'sampling_experiment_manifests' / f'{experiment_name}.json')
                      if resample else model_dir / 'paper_evaluation_manifest.json')
        all_results[model_dir.name] = results
    manifest = {'evaluate_run': str(run), 'mode': 'resample' if resample else 'reuse_saved',
                'explicit_dataset_filter': sorted(real_overrides) if real_overrides else None,
                'evaluated_models': list(all_results), 'excluded_models': excluded,
                'missing_saved_outputs': omitted_outputs, 'results': all_results}
    # A run-level audit makes it easy to verify that a BOG-only invocation did
    # not process other cohorts. Single-model invocation also supports it.
    if resample:
        manifest['experiment_name'] = experiment_name
    _save_summary(manifest, (run / 'sampling_experiment_run_manifests' / f'{experiment_name}.json')
                  if resample else run / 'paper_evaluation_run_manifest.json')
    print(f'[SUMMARY] mode={manifest["mode"]}, evaluated models={list(all_results)}, '
          f'excluded models={len(excluded)}, missing output pairs={len(omitted_outputs)}', flush=True)
    return all_results


def parse_mapping(items):
    out={}
    for s in items:
        if '=' not in s:raise ValueError(f'Expected NAME=PATH: {s}')
        k, p = s.split('=', 1)
        k, p = k.strip().upper(), p.strip()
        if not k or not p:
            raise ValueError(f'Expected NAME=PATH with nonempty name and path: {s}')
        if k in out:
            raise ValueError(f'Duplicate --real_input for {k}; specify one path per dataset')
        out[k] = p
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--evaluate_run',required=True,help='Model directory or whole timestamped run directory')
    parser.add_argument('--real_input',action='append',default=[],metavar='NAME=PATH',
                        help='Explicit dataset filter AND real CSV override. Only models whose datasets are all listed are evaluated; omit to use saved per-model paths.')
    parser.add_argument('--test_input',help='Proven excluded held-out 0/1/2 CSV with identical SNP names')
    parser.add_argument('--resample',action='store_true',
                        help='Regenerate continuous dosages and method-specific genotypes/PCA plots in new sampling_experiments/NAME folder; original outputs remain untouched.')
    parser.add_argument('--experiment_name',default=None,
                        help='Named folder under each checkpoint/sampling_experiments/. Must not already exist; omit for a seed/method-derived name.')
    parser.add_argument('--plots_only',action='store_true',
                        help='With --resample, generate genotype CSVs and diagnostics without costly paper-style evaluations.')
    parser.add_argument('--weights',nargs='+',choices=['baseline','first','best','last'],default=['best','last','baseline','first'])
    parser.add_argument('--sampling_methods',nargs='+',choices=METHODS,default=['binomial'])
    parser.add_argument('--sampling_temperature',type=float,default=.35)
    parser.add_argument('--n_synthetic',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--metric_max_ld_pairs',type=int,default=3000)
    parser.add_argument('--max_distance_samples',type=int,default=256)
    parser.add_argument('--max_motif_windows',type=int,default=256)
    parser.add_argument('--max_triplets',type=int,default=2000)
    args=parser.parse_args()
    evaluate_run(args.evaluate_run,parse_mapping(args.real_input) or None,args.test_input,
                 args.resample,args.sampling_methods,args.weights,args.seed,args.n_synthetic,
                 args.device,args.sampling_temperature,args.metric_max_ld_pairs,
                 args.max_distance_samples,args.max_motif_windows,args.max_triplets,
                 args.experiment_name,args.plots_only)


if __name__=='__main__':main()
