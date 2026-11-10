"""Explicitly requested exploratory fits of nominal labels; never independent scores.
Run directly in a research environment containing scikit-learn, joblib and matplotlib.
No production model loader, database registration, tuning or train/test split.
"""
import argparse,csv,hashlib,json,math,platform
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn,joblib

FEATURE_SETS={
 'compact3':['mask_fraction','long_relative','short_relative'],
 'shape12':['mask_fraction','long_relative','short_relative','aspect_ratio','rect_density','perimeter_normalized','width_p50_relative','width_cv','skeleton_density_512','foreground_L','foreground_b','texture_std'],
 'background_control':['background_mean','background_std','background_bright_fraction','orientation_cos2','image_aspect']}


def models():
    return {'mean':DummyRegressor(strategy='mean'),'linear':make_pipeline(StandardScaler(),LinearRegression()),
      'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
      'random_forest':RandomForestRegressor(n_estimators=100,max_depth=3,min_samples_leaf=2,random_state=42,n_jobs=1),
      'gradient_boosting':GradientBoostingRegressor(n_estimators=50,max_depth=1,learning_rate=.05,random_state=42)}


def metrics(actual,pred):
    actual=np.asarray(actual,float);pred=np.asarray(pred,float)
    if actual.shape!=pred.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(pred).all():raise ValueError('Invalid metrics input')
    e=np.abs(actual-pred)
    return dict(mae_g=float(e.mean()),rmse_g=float(np.sqrt(np.mean(e**2))),max_absolute_error_g=float(e.max()),median_absolute_error_g=float(np.median(e)),mape_pct=float(np.mean(e/actual)*100) if (actual>0).all() else None)


def public_result():
    return dict(status='要確認',predicted_weight_g=None,reason='未承認マスク・名目重量による探索モデル。独立評価未実施。未知製品・範囲外は未対応。')


def validate_independent_assignment(records,assignment):
    """Future explicit verified-lineage split gate. This experiment never passes it."""
    families={};captures={};originals={}
    for r in records:
        if r.get('lineage_verified') is not True or not r.get('physical_family'):raise ValueError('Verified physical lineage required')
        split=assignment[r['image_id']]
        if split not in ('train','validation','test'):raise ValueError('Explicit split required')
        for table,key in [(families,r['physical_family']),(captures,r['capture_group']),(originals,r['source_sha256'])]:
            table.setdefault(key,set()).add(split)
    if any(len(s)>1 for table in (families,captures,originals) for s in table.values()):raise ValueError('Shared source/group crosses split')
    if {s for v in families.values() for s in v}!={'train','validation','test'}:raise ValueError('Independent partitions unavailable')


def run(input_path,output):
    p=Path(input_path);out=Path(output)
    if out.exists():raise FileExistsError('No overwrites')
    source=json.loads(p.read_text());original=[r for r in source['rows'] if r['transform']=='original']
    if source['independent_test_images']!=0 or source['supervised_approval']:raise ValueError('Unexpected approval metadata')
    out.mkdir(parents=True);(out/'models').mkdir();fits=[];predictions=[];perturb=[];permutations=[]
    for method in ('hsv_baseline','hsv_lab'):
        rows=[r for r in original if r['method']==method];y=np.array([r['reported_g'] for r in rows],float)
        if len(rows)!=15 or len({r['capture_group'] for r in rows})!=5:raise ValueError('Expected 15 original observations in 5 capture groups')
        for featureset,keys in FEATURE_SETS.items():
            X=np.array([[r[k] for k in keys] for r in rows],float)
            for name,model in models().items():
                model.fit(X,y);fitted=model.predict(X);ident=f'{method}__{featureset}__{name}'
                # Private artifact is intentionally outside any app model registry.
                joblib.dump(dict(model=model,feature_keys=keys,production_eligible=False,approval='EXPLORATORY_ONLY',nominal_labels=True),out/'models'/(ident+'.joblib'))
                fit=dict(id=ident,mask=method,feature_set=featureset,model=name,n_images=15,n_features=len(keys),design_rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(15),X]))),
                    scope='TRAINING_RESUBSTITUTION_ONLY_NOT_GENERALIZATION',training_metrics=metrics(y,fitted),independent_metrics=None,
                    parameters={k:v for k,v in model.get_params(deep=True).items() if v is None or isinstance(v,(str,int,float,bool))},
                    per_weight={str(int(w)):metrics(y[y==w],fitted[y==w]) for w in np.unique(y)})
                fits.append(fit)
                for r,pred in zip(rows,fitted):predictions.append(dict(id=ident,image_id=r['image_id'],capture_group=r['capture_group'],reported_g=r['reported_g'],training_prediction_g=float(pred),error_g=float(pred-r['reported_g']),public_result=public_result()))
                # These transforms stay tied to their training originals; never test data.
                others=[r for r in source['rows'] if r['method']==method and r['transform']!='original']
                ref={r['image_id']:float(v) for r,v in zip(rows,fitted)}
                for r,v in zip(others,model.predict(np.array([[r[k] for k in keys] for r in others]))):
                    perturb.append(dict(id=ident,image_id=r['image_id'],transform=r['transform'],same_original_group=True,prediction_change_g=float(v-ref[r['image_id']]),research_prediction_g=float(v)))
                if featureset=='shape12':
                    # Preserve all 3 photos' common target when shuffling labels.
                    groups=sorted({r['capture_group'] for r in rows});weights={g:next(r['reported_g'] for r in rows if r['capture_group']==g) for g in groups}
                    for seed in range(10):
                        perm=np.random.default_rng(seed).permutation([weights[g] for g in groups]);mapping=dict(zip(groups,perm));shuffled=np.array([mapping[r['capture_group']] for r in rows])
                        alternative=models()[name];alternative.fit(X,shuffled)
                        permutations.append(dict(id=ident,seed=seed,scope='SHUFFLED_TRAINING_FIT_CAPACITY_ONLY',**metrics(shuffled,alternative.predict(X))))
        print(method,'fits done',flush=True)
    report=dict(purpose='nominal_label_exploratory_fits',training_images=15,approved_production_training_images=0,independent_test_images=0,independent_metrics=None,
        splitting='NONE: all 15 are one unresolved conservative lineage block',tuning='NONE: fixed parameters; no candidate selected for production',
        feature_sets=FEATURE_SETS,versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
        input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fits=fits,predictions=predictions,perturbations=perturb,shuffled_label_fits=permutations,public_result=public_result())
    (out/'models_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    for filename,items in [('training_predictions.csv',predictions),('transform_sensitivity.csv',perturb),('shuffled_fits.csv',permutations)]:
        keys=[k for k in items[0] if k!='public_result']
        with (out/filename).open('w',newline='') as stream:
            wr=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(items)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
