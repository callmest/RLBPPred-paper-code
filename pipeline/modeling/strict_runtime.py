#!/usr/bin/env python3
"""Current strict SaProt-AF2/MLP release: audit, screen, infer, reconstruct.

validate/screen need only Python + NumPy. predict additionally requires PyTorch;
train requires PyTorch and scikit-learn and never runs without --execute.
Frozen release inputs are read-only. All output files/directories must be new.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'data/frozen_model'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def new_output(path, frozen=FROZEN):
    path = Path(path).resolve()
    protected = [Path(frozen).resolve(), ROOT / 'data/model_inputs']
    require(not any(path == p or p in path.parents for p in protected), 'Cannot write into frozen/model-input data')
    require(not path.exists(), f'Output already exists: {path}')
    require(path.parent.is_dir(), f'Output parent does not exist: {path.parent}')
    return path


def write_csv_exclusive(path, fields, rows):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_release(frozen=FROZEN):
    frozen = Path(frozen)
    meta = json.loads((frozen / 'manifest.json').read_text())
    require(meta['schema_version'] == 1, 'Unsupported release manifest')
    for name, item in meta['artifacts'].items():
        require(Path(name).name == name, 'Manifest artifacts must be flat filenames')
        require(digest(frozen / name) == item['sha256'], f'Frozen artifact hash mismatch: {name}')
    return meta


def validate_manifest_rows(manifest, validation, samples):
    ids = [r['protein_id'] for r in manifest]
    require(len(set(ids)) == len(ids) == 9485, 'Strict input IDs must be 9,485 unique proteins')
    require(Counter((r['label'], r['split']) for r in manifest) ==
            {('1', 'train'): 301, ('0', 'train'): 7288,
             ('1', 'validation'): 75, ('0', 'validation'): 1821}, 'Incorrect strict split counts')
    pos = [r['protein_id'] for r in manifest if r['label'] == '1']
    neg = [r['protein_id'] for r in manifest if r['label'] == '0']
    require(ids == sorted(pos) + sorted(neg), 'Frozen input order must be sorted positives then sorted negatives')
    rng = np.random.default_rng(42)
    expected_val = [pos[i] for i in rng.choice(len(pos), 75, replace=False)]
    expected_val += [neg[i] for i in rng.choice(len(neg), 1821, replace=False)]
    require([r['protein_id'] for r in validation] == expected_val, 'Validation ID order differs from actual seed-42 split')
    byid = {r['protein_id']: r for r in manifest}
    require(set(expected_val) == {r['protein_id'] for r in manifest if r['split'] == 'validation'}, 'Validation membership mismatch')
    require(all(r['label'] == byid[r['protein_id']]['label'] for r in validation), 'Validation labels mismatch')
    train = [r for r in manifest if r['split'] == 'train']
    train_ids = [r['protein_id'] for r in train]
    pi = np.array([i for i, r in enumerate(train) if r['label'] == '1'])
    ni = np.array([i for i, r in enumerate(train) if r['label'] == '0'])
    rng = np.random.default_rng(42)
    require(len(samples) == 300, 'Expected 300 sampling rounds')
    for i, row in enumerate(samples):
        selected = rng.choice(ni, size=301, replace=False)
        order = np.concatenate([pi, selected]); rng.shuffle(order)
        require(row['round'] == i + 1, 'Incorrect sampling round number')
        require(row['negative_ids'] == [train_ids[j] for j in selected], f'Round {i+1}: negative sample differs')
        require(row['training_order'] == [train_ids[j] for j in order], f'Round {i+1}: training order differs')
    return ids


def screening_summary(validation, human, threshold=0.98):
    require(0 <= threshold <= 1, 'Threshold must be in [0,1]')
    negatives = [float(r['probability']) for r in validation if r['label'] == '0']
    pool = [r for r in human if r['dataset'] == 'untrained_proteome']
    require(len(negatives) == 1821 and len(pool) == 5640, 'Incorrect held-out/discovery denominators')
    candidates = [r for r in pool if float(r['probability']) >= threshold]
    candidates.sort(key=lambda r: (-float(r['probability']), r['protein_id']))
    fp = sum(p >= threshold for p in negatives)
    return candidates, {'threshold': threshold, 'operator': '>=', 'false_positives': fp,
                        'heldout_negative_n': 1821, 'empirical_fpr': fp / 1821,
                        'target_empirical_fpr': 0.001, 'target_satisfied': fp / 1821 <= 0.001,
                        'candidate_count': len(candidates), 'untrained_scored_n': 5640}


def validate_release(frozen=FROZEN):
    frozen = Path(frozen); meta = load_release(frozen)
    manifest = read_csv(frozen / 'strict_input_manifest.csv')
    validation = read_csv(frozen / 'val_predictions.csv')
    samples = json.loads((frozen / 'sampling_manifest.json').read_text())
    ids = validate_manifest_rows(manifest, validation, samples)
    human = read_csv(frozen / 'human_all_predictions.csv')
    mouse = read_csv(frozen / 'mouse_all_predictions.csv')
    require(Counter(r['dataset'] for r in human) == {'sig_pos_ids':376,'neg_ids':9109,'potential_positive':1844,'untrained_proteome':5640}, 'Incorrect human coverage')
    require(Counter(r['dataset'] for r in mouse) == {'significant_positive':76,'potential_positive':284,'negative':16642}, 'Incorrect mouse coverage')
    for label, rows in [('human', human), ('mouse', mouse), ('validation', validation)]:
        require(len({r['protein_id'] for r in rows}) == len(rows), f'Duplicate {label} IDs')
        require(all(np.isfinite(float(r['probability'])) and 0 <= float(r['probability']) <= 1 for r in rows), f'Invalid {label} scores')
    byid = {r['protein_id']: r for r in human}
    require(all(float(r['probability']) == float(byid[r['protein_id']]['probability']) for r in validation), 'Validation/full-human score mismatch')
    pool = {r['protein_id'] for r in human if r['dataset'] == 'untrained_proteome'}
    require(not pool.intersection(ids), 'Discovery pool overlaps strict training or validation')
    candidates, screening = screening_summary(validation, human)
    frozen_candidates = read_csv(frozen / 'candidates_401.csv')
    require(len(candidates) == 401 and screening['false_positives'] == 1, 'Current screening result mismatch')
    require({r['protein_id']: float(r['probability']) for r in candidates} ==
            {r['protein_id']: float(r['probability']) for r in frozen_candidates}, 'Candidate scores/IDs mismatch')
    discrepancy = max(abs(float(r['probability'])-float(r['probability_L40S'])) for r in frozen_candidates)
    require(discrepancy <= meta['cross_machine_probability_tolerance'], 'Candidate cross-machine discrepancy exceeds tolerance')
    inputs = ROOT / 'data/model_inputs'
    if inputs.is_dir():
        for name, expected in [('significant_positive_ids.txt',{r['protein_id'] for r in manifest if r['label']=='1'}),
                               ('negative_ids.txt',{r['protein_id'] for r in manifest if r['label']=='0'}),
                               ('high_confidence_candidate_ids.txt',{r['protein_id'] for r in candidates})]:
            lines=(inputs/name).read_text().split(); require(len(lines)==len(set(lines)) and set(lines)==expected, f'ID list mismatch: {name}')
    return {'pass':True,'run_id':meta['run_id'],'strict_inputs':9485,'ensemble_rounds':300,
            'human_scored':len(human),'mouse_scored':len(mouse),'screening':screening,
            'candidate_cross_machine_max_abs_difference':discrepancy}


def validate_checkpoint(ck):
    require(isinstance(ck,Mapping), 'Checkpoint must be a mapping')
    expected = {'format':'state_dict_ensemble_v1','upstream':'saprot-650m-af2','clf_name':'MLP',
                'in_dim':1280,'hidden_dims':[512,256],'dropout':0.3,'use_bn':True,
                'n_rounds':300,'seed':42,'scaler_n_samples_seen':7589}
    for key, value in expected.items():require(ck.get(key)==value, f'Checkpoint mismatch: {key}')
    states=ck.get('model_state_dicts')
    require(isinstance(states,(list,tuple)) and len(states)==300, 'Checkpoint must contain 300 states')

    def numeric_array(value, name):
        # Duck-typing avoids importing torch for release audits/unit tests.
        # Real checkpoints are loaded on CPU; detach supports requires_grad tensors.
        try:
            if hasattr(value,'detach'):value=value.detach()
            if hasattr(value,'cpu'):value=value.cpu()
            if hasattr(value,'numpy'):value=value.numpy()
            array=np.asarray(value)
        except (TypeError,ValueError,RuntimeError) as exc:
            raise ValueError(f'Cannot inspect {name} as a numeric array') from exc
        require(array.dtype.kind in 'fiu', f'{name} must have a real numeric dtype')
        require(np.isfinite(array).all(), f'{name} contains non-finite values')
        return array

    for key in ['scaler_mean','scaler_scale','scaler_var']:
        arr=numeric_array(ck.get(key),key);require(arr.shape==(1280,), f'Invalid {key} shape')
        if key=='scaler_scale':require((arr>0).all(), 'Scaler scales must be positive')
        if key=='scaler_var':require((arr>=0).all(), 'Scaler variances must be nonnegative')
    shapes={'net.0.weight':(512,1280),'net.0.bias':(512,),
            'net.4.weight':(256,512),'net.4.bias':(256,),
            'net.8.weight':(1,256),'net.8.bias':(1,)}
    for layer,width in [(1,512),(5,256)]:
        for field in ['weight','bias','running_mean','running_var']:
            shapes[f'net.{layer}.{field}']=(width,)
        shapes[f'net.{layer}.num_batches_tracked']=()
    for member,state in enumerate(states,1):
        require(isinstance(state,Mapping),f'Ensemble member {member} must be a state mapping')
        require(set(state)==set(shapes),f'Ensemble member {member}: missing or unexpected state keys')
        for key,shape in shapes.items():
            label=f'Ensemble member {member}: {key}'
            arr=numeric_array(state[key],label)
            require(arr.shape==shape,f'{label} has shape {arr.shape}, expected {shape}')
            if key.endswith('num_batches_tracked'):
                require(arr.dtype.kind in 'iu' and int(arr)==500,f'{label} must be integer 500 (50 epochs x 10 batches)')
            else:
                require(arr.dtype==np.dtype('float32'),f'{label} must be float32')
                if key.endswith('running_var'):require((arr>=0).all(),f'{label} must be nonnegative')
    return ck


def load_checkpoint(path, expected_hash, torch_module=None):
    # This historical state-dict archive includes NumPy objects. Verify its SHA256
    # before deserializing with weights_only=False; never load an arbitrary pickle.
    require(digest(path)==expected_hash, 'Checkpoint SHA256 mismatch; deserialization refused')
    if torch_module is None:
        import torch as torch_module
    return validate_checkpoint(torch_module.load(path,map_location='cpu',weights_only=False))


def scale_float32(raw, ck):
    raw=np.array(raw,dtype=np.float32,copy=True)
    require(raw.ndim==2 and raw.shape[1]==1280 and np.isfinite(raw).all(), 'Expected finite N x 1280 embeddings')
    # Same in-place float32 rounding as the actual sklearn StandardScaler transform.
    raw-=np.asarray(ck['scaler_mean']); raw/=np.asarray(ck['scaler_scale'])
    return raw


def build_mlp(torch, device):
    """Architecture only; avoids importing sklearn/matplotlib for inference."""
    class StrictMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers=[]; previous=1280
            for hidden in [512,256]:
                layers.extend([torch.nn.Linear(previous,hidden),torch.nn.BatchNorm1d(hidden),
                               torch.nn.ReLU(),torch.nn.Dropout(.3)])
                previous=hidden
            layers.append(torch.nn.Linear(previous,1))
            self.net=torch.nn.Sequential(*layers)
        def forward(self,x):return self.net(x).squeeze(-1)
    return StrictMLP().to(device)


def predict(args):
    import torch
    meta=load_release(args.frozen);output=new_output(args.output,args.frozen)
    ck=load_checkpoint(args.model,args.model_sha256 or meta['checkpoint']['sha256'])
    with np.load(args.input,allow_pickle=False) as data:
        ids=np.asarray(data['protein_id']).astype(str);raw=np.array(data['X'],dtype=np.float32,copy=True)
    require(ids.ndim==1 and len(ids)>0, 'Embedding IDs must be a nonempty 1-D array')
    require(raw.shape==(len(ids),1280) and len(set(ids))==len(ids), 'Embedding IDs/shape are invalid')
    X=scale_float32(raw,ck);torch.set_num_threads(2)
    model=build_mlp(torch,args.device);model.eval()
    total=np.zeros(len(ids))
    with torch.no_grad():
        for state in ck['model_state_dicts']:
            model.load_state_dict(state,strict=True)
            for start in range(0,len(ids),args.batch_size):
                logits=model(torch.tensor(X[start:start+args.batch_size],dtype=torch.float32,device=args.device)).cpu().numpy()
                total[start:start+len(logits)]+=1/(1+np.exp(-logits))
    probs=total/300
    require(np.isfinite(probs).all(),'Non-finite predicted scores')
    write_csv_exclusive(output,['protein_id','probability','classification_at_0_5'],
                        [{'protein_id':i,'probability':float(p),'classification_at_0_5':int(p>=.5)} for i,p in zip(ids,probs)])
    return {'output':str(output),'n':len(ids),'model_sha256':digest(args.model)}


def train(args):
    """Replay the actual strict protocol; do not add a new reseeding schedule."""
    require(args.execute,'Training is disabled unless --execute is explicitly supplied')
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from train_sa_mlp import MLPClassifier
    validate_release(args.frozen);frozen=Path(args.frozen);out=new_output(args.output_dir,args.frozen)
    manifest=read_csv(frozen/'strict_input_manifest.csv');val=read_csv(frozen/'val_predictions.csv')
    # Identical process-level seeding to the frozen 5880 run, before loading data.
    torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    emb=Path(args.embedding_dir);index={}
    for label,folder in [('1','positive_mean'),('0','negative_mean')]:
        paths=sorted((emb/folder).glob('*.pt')); expected={r['protein_id'] for r in manifest if r['label']==label}
        require({p.stem for p in paths}==expected,f'Missing/unexpected strict embeddings in {folder}')
        index.update({p.stem:p for p in paths})
    X=np.array([torch.load(index[r['protein_id']],map_location='cpu',weights_only=False)['embeddings'].float().numpy() for r in manifest])
    require(X.shape==(9485,1280) and X.dtype==np.float32 and np.isfinite(X).all(),'Incorrect embedding matrix')
    expected=json.loads((frozen/'input_validation.json').read_text())['embedding_matrix_sha256']
    require(hashlib.sha256(X.tobytes()).hexdigest()==expected,'Strict embedding matrix hash mismatch')
    positions={r['protein_id']:i for i,r in enumerate(manifest)}
    ti=[i for i,r in enumerate(manifest) if r['split']=='train']
    vi=[positions[r['protein_id']] for r in val]
    labels=np.array([int(r['label']) for r in manifest]);scaler=StandardScaler()
    xt=scaler.fit_transform(X[ti]);xv=scaler.transform(X[vi]);yt=labels[ti]
    trainpos={manifest[i]['protein_id']:j for j,i in enumerate(ti)}
    samples=json.loads((frozen/'sampling_manifest.json').read_text());total=np.zeros(len(vi));states=[]
    out.mkdir();(out/'checkpoints').mkdir()
    for sample in samples:
        idx=[trainpos[i] for i in sample['training_order']]
        clf=MLPClassifier(1280,hidden_dims=(512,256),dropout=.3,use_bn=True,device=args.device)
        clf.fit(xt[idx],yt[idx]);total+=clf.predict_proba(xv)[:,1]
        state={k:v.detach().cpu().clone() for k,v in clf.model.state_dict().items()}
        require(int(state['net.1.num_batches_tracked'])==500,'Unexpected training epoch/batch count')
        states.append(state);torch.save(state,out/'checkpoints'/f"round_{sample['round']:03d}.pt")
        del clf
    ck=dict(format='state_dict_ensemble_v1',upstream='saprot-650m-af2',clf_name='MLP',in_dim=1280,
            hidden_dims=[512,256],dropout=.3,use_bn=True,n_rounds=300,seed=42,model_state_dicts=states,
            scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,scaler_var=scaler.var_,scaler_n_samples_seen=int(scaler.n_samples_seen_))
    validate_checkpoint(ck);torch.save(ck,out/'strict_ensemble.pt');probs=total/300
    write_csv_exclusive(out/'val_predictions.csv',['protein_id','label','probability'],
                        [dict(protein_id=r['protein_id'],label=r['label'],probability=float(p)) for r,p in zip(val,probs)])
    result={'auroc':float(roc_auc_score(labels[vi],probs)),'auprc':float(average_precision_score(labels[vi],probs)),
            'confusion_matrix_at_0_5':confusion_matrix(labels[vi],probs>=.5).tolist(),
            'reference_prediction_max_abs_difference':float(np.max(np.abs(probs-np.array([float(r['probability']) for r in val])))),
            'model_sha256':digest(out/'strict_ensemble.pt'),'torch':torch.__version__,'numpy':np.__version__,
            'note':'Frozen split/order/sampling and RNG protocol replayed. Hardware/library differences can change scores; no assertion of byte-identical training.'}
    with (out/'reconstruction_report.json').open('x') as handle:json.dump(result,handle,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=FROZEN)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='Verify frozen hashes, split, 300-round samples, probabilities and 401 candidates')
    screen=sub.add_parser('screen',help='Screen only frozen untrained-pool scores; FPR uses held-out negatives')
    screen.add_argument('--output',type=Path,required=True)
    pred=sub.add_parser('predict',help='Load trusted state-dict ensemble and score NPZ X/protein_id')
    pred.add_argument('--model',type=Path,required=True);pred.add_argument('--model-sha256',help='Explicit SHA256 for a trusted reconstruction; otherwise the published frozen hash is required')
    pred.add_argument('--input',type=Path,required=True);pred.add_argument('--output',type=Path,required=True)
    pred.add_argument('--device',choices=['cpu','cuda'],default='cpu');pred.add_argument('--batch-size',type=int,default=2048)
    tr=sub.add_parser('train',help='Opt-in strict reconstruction; never run for figure regeneration')
    tr.add_argument('--embedding-dir',type=Path,required=True);tr.add_argument('--output-dir',type=Path,required=True)
    tr.add_argument('--device',choices=['cpu','cuda'],default='cuda');tr.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.command=='validate':result=validate_release(args.frozen)
    elif args.command=='screen':
        validate_release(args.frozen);output=new_output(args.output,args.frozen)
        candidates,result=screening_summary(read_csv(args.frozen/'val_predictions.csv'),read_csv(args.frozen/'human_all_predictions.csv'))
        write_csv_exclusive(output,['dataset','protein_id','probability'],candidates);result['output']=str(output)
    elif args.command=='predict':
        require(args.batch_size>0,'Batch size must be positive');result=predict(args)
    else:result=train(args)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
