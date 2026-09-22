"""Inference audit for Synthetic Geometry Classifiers. Apache-2.0."""
import os,json,csv,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
root=Path(os.environ.get('MODEL_ROOT','/kaggle/input'))
paths=sorted(root.rglob('model.json'))
assert paths, f'No model exports found in {root}'
def predict(x,m):
 x=np.asarray(x,dtype=float)
 if x.ndim!=2 or x.shape[1]!=2:raise ValueError('Expected N-by-2 coordinates')
 if not np.isfinite(x).all() or (np.abs(x)>1).any():raise ValueError('Coordinates outside domain')
 v=m['variant']
 if v=='linear':f=x
 elif v=='quadratic':f=np.column_stack([x,x[:,0]**2,x[:,0]*x[:,1],x[:,1]**2])
 elif v=='radial':f=(x*x).sum(axis=1).reshape(-1,1)
 else:raise ValueError(v)
 z=f@np.asarray(m['coefficients'])+m['intercept']
 return 1/(1+np.exp(-np.clip(z,-700,700)))
rng=np.random.default_rng(1203);x=rng.uniform(-1,1,(2000,2));clean=((x*x).sum(axis=1)<.5).astype(int);y=np.where(rng.random(2000)<.05,1-clean,clean)
models={};rows=[]
for path in paths:
 m=json.loads(path.read_text());v=json.loads(path.with_name('test_vectors.json').read_text())
 np.testing.assert_allclose(predict(v['inputs'],m),v['probability_class_1'],atol=1e-12,rtol=1e-12)
 if m['variant'] in models:
  prior=models[m['variant']]
  assert {k:v for k,v in m.items() if k not in ('coefficients','intercept')}=={k:v for k,v in prior.items() if k not in ('coefficients','intercept')}
  np.testing.assert_allclose(m['coefficients'],prior['coefficients'],atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(m['intercept'],prior['intercept'],atol=1e-12,rtol=1e-12)
  continue
 models[m['variant']]=m;p=predict(x,m);saved=json.loads(path.with_name('metrics.json').read_text())
 scores={'variant':m['variant'],'accuracy':accuracy_score(y,p>=m['threshold']),'log_loss':log_loss(y,p),'roc_auc':roc_auc_score(y,p)}
 for k in ['accuracy','log_loss','roc_auc']:np.testing.assert_allclose(scores[k],saved['test_'+k],atol=1e-12,rtol=1e-12)
 rows.append(scores)
assert set(models)=={'linear','quadratic','radial'}
for bad in [[[2,0]],[[float('nan'),0]],[0,0]]:
 try:predict(bad,models['radial'])
 except ValueError:pass
 else:raise AssertionError('Invalid input accepted')
assert predict(np.empty((0,2)),models['radial']).shape==(0,)
with open('inference_audit.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2));print(f'PASS: {len(paths)} exports verified; three variants reproduce held-out metrics without training')
try:
 import matplotlib.pyplot as plt
 grid=np.linspace(-1,1,101);a,b=np.meshgrid(grid,grid);points=np.column_stack([a.ravel(),b.ravel()])
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,(name,m) in zip(axes,sorted(models.items())):
  im=ax.imshow(predict(points,m).reshape(a.shape),origin='lower',extent=(-1,1,-1,1),vmin=0,vmax=1,cmap='viridis');ax.set_title(name);ax.set_xlabel('x1');ax.set_ylabel('x2')
 fig.colorbar(im,ax=axes,label='Probability of noisy class 1');fig.savefig('decision_surfaces.png',bbox_inches='tight');plt.show()
except ImportError:print('Optional plot skipped: matplotlib unavailable')
