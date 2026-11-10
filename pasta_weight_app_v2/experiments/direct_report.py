"""Render descriptive graphs and tables from private direct-weight experiments."""
import argparse,csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root):
    root=Path(root);out=root/'analysis'
    if out.exists():raise FileExistsError('No overwrites')
    f=json.loads((root/'features/features.json').read_text());m=json.loads((root/'models_experiment/models_report.json').read_text());out.mkdir()
    rows=[r for r in f['rows'] if r['transform']=='original'];methods=['hsv_baseline','hsv_lab']
    keys=['mask_area_px','mask_fraction','long_relative','short_relative','width_p50_relative','rect_density','skeleton_density_512','texture_std','background_mean']
    stats=[];fig,axs=plt.subplots(3,3,figsize=(15,12))
    for ax,key in zip(axs.flat,keys):
        for method,color,offset in zip(methods,['#b76e50','#43775d'],[-1,1]):
            for weight in (20,40,60,80,100):
                values=[r[key] for r in rows if r['method']==method and r['reported_g']==weight]
                ax.scatter([weight+offset]*3,values,color=color,s=20,label=method if weight==20 else None)
                ax.plot([weight+offset]*2,[min(values),max(values)],color=color,alpha=.5)
                stats.append(dict(method=method,reported_g=weight,feature=key,n=3,minimum=min(values),maximum=max(values),mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),cv=float(np.std(values,ddof=1)/abs(np.mean(values))) if np.mean(values) else None))
        ax.set_title(key);ax.set_xlabel('Nominal g (same capture group at each weight)');ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('Within-weight arrangement variation | unresolved physical lineage | no accuracy claim');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'feature_variation.png',dpi=140);plt.close(fig)
    with (out/'within_group_variation.csv').open('w',newline='') as stream:
        wr=csv.DictWriter(stream,fieldnames=list(stats[0]));wr.writeheader();wr.writerows(stats)
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for i,method in enumerate(methods):
        for j,fs in enumerate(['compact3','shape12','background_control']):
            fits=[x for x in m['fits'] if x['mask']==method and x['feature_set']==fs];ax=axs[i,j]
            ax.bar([x['model'] for x in fits],[x['training_metrics']['mae_g'] for x in fits],color='#43775d');ax.tick_params(axis='x',rotation=30);ax.set_ylabel('Training MAE (g)');ax.set_title(method+' / '+fs);ax.grid(axis='y',alpha=.2)
    fig.suptitle('TRAINING FIT ONLY | 15 images, no independent test | smaller is not verified better');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'training_fit.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,6));fits=[x for x in m['fits'] if x['mask']=='hsv_lab' and x['feature_set']=='shape12']
    for i,fit in enumerate(fits):
        vals=[x['mae_g'] for x in m['shuffled_label_fits'] if x['id']==fit['id']];axs[0].scatter([i]*len(vals),vals,alpha=.4,color='#b76e50');axs[0].scatter(i,fit['training_metrics']['mae_g'],color='#264a3b',s=70,marker='D')
        for transform,offset,color in [('resize_half',-.17,'#43775d'),('rotate90',.17,'#b76e50')]:
            vals=[abs(x['prediction_change_g']) for x in m['perturbations'] if x['id']==fit['id'] and x['transform']==transform]
            axs[1].bar(i+offset,max(vals),width=.3,color=color,label=transform if i==0 else None)
    for ax in axs:ax.set_xticks(range(5),[x['model'] for x in fits],rotation=25);ax.grid(axis='y',alpha=.2)
    axs[0].set_title('Shuffled group labels: dots; actual training fit: diamonds');axs[0].set_ylabel('Training MAE (g), not test error');axs[1].set_title('Same-original transform sensitivity');axs[1].set_ylabel('Maximum absolute prediction change (g)');axs[1].legend();fig.suptitle('Overfitting capacity and instability | HSV+Lab / shape12');fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'capacity_stability.png',dpi=140);plt.close(fig)
    cols=['image_id','reported_g','method','mask_area_px','mask_fraction','long_axis_px','short_axis_px','orientation_deg','width_p50_relative','contour_perimeter_px','components','skeleton_density_512','legacy_candidate_count','qc_flags']
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>方式B・探索的比較</title><style>body{font-family:sans-serif;background:#faf8f2;color:#29352e;margin:20px}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;background:white}td,th{border:1px solid #ddd;padding:7px;white-space:nowrap}select{padding:10px}</style><h1>方式B：実験結果</h1><p>15枚・5撮影群、物理試料の独立性は未確認。すべて訓練内の探索的当てはまりです。正式な検証MAE/RMSEは未算出。数値を一般利用者の重量推定へ適用しません。</p><p><a href="../REPORT_JA.md">日本語報告</a> · <a href="../features/index.html">原本・マスク比較</a> · <a href="within_group_variation.csv">同重量の特徴量変動CSV</a> · <a href="../models_experiment/training_predictions.csv">訓練内予測CSV</a> · <a href="../models_experiment/transform_sensitivity.csv">変換への反応CSV</a></p><h2>モデル比較（訓練内のみ）</h2><div class="scroll"><table><tr><th>マスク</th><th>特徴量</th><th>モデル</th><th>訓練MAE g</th><th>訓練RMSE g</th><th>訓練最大誤差 g</th><th>独立検証</th></tr>'''
    for fit in m['fits']:
        v=fit['training_metrics'];page+=f'<tr><td>{fit["mask"]}</td><td>{fit["feature_set"]}</td><td>{fit["model"]}</td><td>{v["mae_g"]:.2f}</td><td>{v["rmse_g"]:.2f}</td><td>{v["max_absolute_error_g"]:.2f}</td><td>未算出</td></tr>'
    page+='</table></div><h2>重量別・画像別特徴量</h2><label>申告重量 <select id="weight"><option value="all">すべて</option>'+''.join(f'<option>{w}</option>' for w in (20,40,60,80,100))+'</select></label><div class="scroll"><table><tr>'+''.join(f'<th>{k}</th>' for k in cols)+'</tr>'
    for row in rows:page+=f'<tr data-weight="{row["reported_g"]}">'+''.join('<td>'+html.escape(f'{row[k]:.4g}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in cols)+'</tr>'
    page+='</table></div>'+''.join(f'<h2>{title}</h2><a href="{file}"><img src="{file}"></a>' for file,title in [('feature_variation.png','同重量の配置差'),('training_fit.png','モデル比較：訓練誤差のみ'),('capacity_stability.png','過学習の検討・縮小と回転への反応')])
    page+='''<script>document.getElementById('weight').onchange=e=>document.querySelectorAll('[data-weight]').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.weight!==e.target.value)</script></html>'''
    (out/'index.html').write_text(page)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
