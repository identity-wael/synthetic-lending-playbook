"""Educational synthetic geometry classifiers. Apache-2.0; Codex-assisted."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
OUT=Path('model_exports');OUT.mkdir(exist_ok=True)
SEEDS={'train':1201,'validation':1202,'test':1203}
def data(seed,n):
 r=np.random.default_rng(seed);x=r.uniform(-1,1,(n,2));clean=(np.sum(x*x,axis=1)<0.5).astype(int)
 return x,np.where(r.random(n)<0.05,1-clean,clean)
def features(x,v):
 if v=='linear':return x
 if v=='quadratic':return np.column_stack([x,x[:,0]**2,x[:,0]*x[:,1],x[:,1]**2])
 if v=='radial':return np.sum(x*x,axis=1).reshape(-1,1)
 raise ValueError(v)
def predict_json(x,m):
 z=features(x,m['variant'])@np.asarray(m['coefficients'])+m['intercept']
 return 1/(1+np.exp(-np.clip(z,-700,700)))
splits={k:data(s,6000 if k=='train' else 2000) for k,s in SEEDS.items()}
assert len({hashlib.sha256(x.tobytes()).hexdigest() for x,y in splits.values()})==3
scores=[]
for v in ['linear','quadratic','radial']:
 candidates=[]
 for C in [0.1,1.,10.]:
  fit=LogisticRegression(C=C,max_iter=1000,solver='lbfgs').fit(features(splits['train'][0],v),splits['train'][1])
  loss=log_loss(splits['validation'][1],fit.predict_proba(features(splits['validation'][0],v))[:,1]);candidates.append((loss,C,fit))
 val,C,fit=min(candidates,key=lambda t:t[0]);folder=OUT/v;folder.mkdir(exist_ok=True)
 m={'schema_version':1,'variant':v,'coefficients':fit.coef_[0].tolist(),'intercept':float(fit.intercept_[0]),'classes':[0,1],'threshold':0.5,'selected_C':C,'input_order':['x1','x2'],'input_domain':'Each coordinate in [-1,1]','label':'Inside radius sqrt(0.5), with independent 5% label flips','feature_order':{'linear':['x1','x2'],'quadratic':['x1','x2','x1^2','x1*x2','x2^2'],'radial':['x1^2+x2^2']}[v]}
 (folder/'model.json').write_text(json.dumps(m,indent=2));loaded=json.loads((folder/'model.json').read_text());x,y=splits['test'];p=predict_json(x,loaded)
 np.testing.assert_allclose(p,fit.predict_proba(features(x,v))[:,1],atol=1e-12,rtol=1e-12)
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 metrics={'variant':v,'selected_C':C,'validation_log_loss':val,'test_accuracy':accuracy_score(y,p>=0.5),'test_log_loss':log_loss(y,p),'test_roc_auc':roc_auc_score(y,p),'test_n':len(y),'positive_fraction':float(y.mean()),'majority_accuracy':float(max(y.mean(),1-y.mean()))}
 (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));scores.append(metrics)
 (folder/'test_vectors.json').write_text(json.dumps({'inputs':x[:8].tolist(),'probability_class_1':p[:8].tolist()},indent=2))
 (folder/'README.md').write_text(f'''# Synthetic Geometry: {v}\n\nEducational binary logistic classifier. Apache-2.0. Created by Wael El Ghazzawi with OpenAI Codex assistance.\n\nInputs x1,x2 are uniform coordinates in [-1,1]. Label 1 means x1^2+x2^2 < 0.5, followed by independent 5% label flips. No validated real-world use.\n\nTrain:6000 points seed1201; validation:2000 seed1202; test:2000 seed1203. Select C among 0.1,1,10 by validation log loss, never test data. Metrics apply only to this noisy generator, without confidence intervals or external validation.\n\nJSON contains explicit feature order, coefficients and intercept. Transform inputs, take the coefficient dot product plus intercept, apply sigmoid, then threshold at 0.5. Test vectors verify other implementations. No pickle or executable weights. No fine-tuning API; retrain using source. Version1.\n''')
manifest={'seeds':SEEDS,'sample_sizes':{k:len(y) for k,(x,y) in splits.items()},'numpy':np.__version__,'sklearn':sklearn.__version__,'label_noise':0.05,'split_sha256':{k:hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for k,(x,y) in splits.items()}}
(OUT/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
print(json.dumps(scores,indent=2));print('PASS: all JSON exports reproduce sklearn probabilities on 2000 held-out samples')
