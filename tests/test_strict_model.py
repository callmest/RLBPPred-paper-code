"""Small read-only release/loader tests. No fitting, training or model download."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('strict_runtime',ROOT/'pipeline/modeling/strict_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)


def checkpoint():
    # Shape-correct NumPy stand-ins, shared read-only arrays keep the fixture small.
    shapes={'net.0.weight':(512,1280),'net.0.bias':(512,),
            'net.4.weight':(256,512),'net.4.bias':(256,),
            'net.8.weight':(1,256),'net.8.bias':(1,)}
    for layer,width in [(1,512),(5,256)]:
        for field in ['weight','bias','running_mean','running_var']:
            shapes[f'net.{layer}.{field}']=(width,)
    state={key:np.broadcast_to(np.float32(1 if key.endswith('running_var') else 0),shape)
           for key,shape in shapes.items()}
    state.update({f'net.{layer}.num_batches_tracked':np.array(500,dtype=np.int64) for layer in [1,5]})
    return dict(format='state_dict_ensemble_v1',upstream='saprot-650m-af2',clf_name='MLP',
                in_dim=1280,hidden_dims=[512,256],dropout=.3,use_bn=True,n_rounds=300,
                seed=42,scaler_n_samples_seen=7589,model_state_dicts=[dict(state) for _ in range(300)],
                scaler_mean=np.zeros(1280),scaler_scale=np.ones(1280),scaler_var=np.ones(1280))


class StrictModelTests(unittest.TestCase):
    def test_frozen_release_and_all_300_rounds(self):
        result=runtime.validate_release()
        self.assertTrue(result['pass']);self.assertEqual(result['screening']['candidate_count'],401)
        self.assertEqual(result['screening']['false_positives'],1)
        self.assertEqual(result['screening']['heldout_negative_n'],1821)

    def test_split_order_change_rejected(self):
        m=runtime.read_csv(runtime.FROZEN/'strict_input_manifest.csv')
        v=runtime.read_csv(runtime.FROZEN/'val_predictions.csv')
        s=json.loads((runtime.FROZEN/'sampling_manifest.json').read_text())
        v[0],v[1]=v[1],v[0]
        with self.assertRaisesRegex(ValueError,'Validation ID order'):runtime.validate_manifest_rows(m,v,s)

    def test_sampling_change_rejected(self):
        m=runtime.read_csv(runtime.FROZEN/'strict_input_manifest.csv')
        v=runtime.read_csv(runtime.FROZEN/'val_predictions.csv')
        s=json.loads((runtime.FROZEN/'sampling_manifest.json').read_text())
        s[0]['negative_ids'].reverse()
        with self.assertRaisesRegex(ValueError,'negative sample'):runtime.validate_manifest_rows(m,v,s)

    def test_checkpoint_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';p.write_bytes(b'synthetic fixture, not a pickle')
            loader=SimpleNamespace(load=lambda *a,**k:self.fail('Should not deserialize a wrong hash'))
            with self.assertRaisesRegex(ValueError,'SHA256'):runtime.load_checkpoint(p,'0'*64,loader)
            calls=[]
            loader=SimpleNamespace(load=lambda *a,**k:(calls.append((a,k)) or checkpoint()))
            ck=runtime.load_checkpoint(p,runtime.digest(p),loader)
            self.assertEqual(len(ck['model_state_dicts']),300)
            self.assertFalse(calls[0][1]['weights_only'])

    def test_checkpoint_architecture_and_scaler_rejected(self):
        ck=checkpoint();ck['n_rounds']=299
        with self.assertRaises(ValueError):runtime.validate_checkpoint(ck)
        ck=checkpoint();ck['scaler_scale'][0]=0
        with self.assertRaisesRegex(ValueError,'positive'):runtime.validate_checkpoint(ck)

    def test_empty_missing_and_extra_state_keys_rejected(self):
        for mutation in ['empty','missing','extra']:
            with self.subTest(mutation=mutation):
                ck=checkpoint();state=ck['model_state_dicts'][0]
                if mutation=='empty':state.clear()
                elif mutation=='missing':del state['net.0.bias']
                else:state['unexpected']=np.array(0,dtype=np.float32)
                with self.assertRaisesRegex(ValueError,'member 1: missing or unexpected'):
                    runtime.validate_checkpoint(ck)

    def test_last_member_bad_shape_and_nonfinite_rejected(self):
        cases=[('net.8.bias',np.zeros(2,dtype=np.float32),'shape'),
               ('net.8.bias',np.array([np.nan],dtype=np.float32),'non-finite'),
               ('net.5.running_mean',np.full(256,np.inf,dtype=np.float32),'non-finite'),
               ('net.5.running_var',np.full(256,-1,dtype=np.float32),'nonnegative'),
               ('net.8.bias',np.zeros(1,dtype=np.float64),'float32'),
               ('net.1.num_batches_tracked',np.array(0,dtype=np.int64),'integer 500')]
        for key,value,message in cases:
            with self.subTest(key=key,message=message):
                ck=checkpoint();ck['model_state_dicts'][299][key]=value
                with self.assertRaisesRegex(ValueError,f'member 300: .*{message}'):
                    runtime.validate_checkpoint(ck)

    def test_tensor_like_state_inspected_without_torch_import(self):
        calls=[]
        class TensorLike:
            def detach(self):calls.append('detach');return self
            def cpu(self):calls.append('cpu');return self
            def numpy(self):calls.append('numpy');return np.zeros(1,dtype=np.float32)
        ck=checkpoint();ck['model_state_dicts'][299]['net.8.bias']=TensorLike()
        self.assertIs(runtime.validate_checkpoint(ck),ck)
        self.assertEqual(calls,['detach','cpu','numpy'])

    def test_float32_scaling_preserves_original_and_rounding(self):
        raw=np.arange(2560,dtype=np.float32).reshape(2,1280)/7
        before=raw.copy();ck=checkpoint();ck['scaler_mean']=np.linspace(.01,.9,1280);ck['scaler_scale']=np.linspace(.2,2,1280)
        expected=raw.copy();expected-=ck['scaler_mean'];expected/=ck['scaler_scale']
        actual=runtime.scale_float32(raw,ck)
        np.testing.assert_array_equal(raw,before);np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual.dtype,np.float32)

    def test_screen_operator_and_heldout_scope(self):
        v=[{'label':'0','probability':.98}]+[{'label':'0','probability':0.0} for _ in range(1820)]
        h=[{'protein_id':str(i),'dataset':'untrained_proteome','probability':.98 if i<401 else .97} for i in range(5640)]
        h += [{'protein_id':'training-negative','dataset':'neg_ids','probability':1.0}]
        c,s=runtime.screening_summary(v,h)
        self.assertEqual(len(c),401);self.assertEqual(s['false_positives'],1)
        self.assertEqual(s['empirical_fpr'],1/1821)

    def test_output_protection_and_training_opt_in(self):
        with self.assertRaises(ValueError):runtime.new_output(runtime.FROZEN/'new.csv')
        with self.assertRaisesRegex(ValueError,'disabled'):runtime.train(SimpleNamespace(execute=False))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.csv';p.touch()
            with self.assertRaisesRegex(ValueError,'exists'):runtime.new_output(p)


if __name__=='__main__':unittest.main()
