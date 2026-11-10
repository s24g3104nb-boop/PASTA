"""Offline paired-stability figures, CSV and image/mask/prediction inspection pages."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .stability_models import csv_write

STYLE = '<style>body{font-family:sans-serif;background:#faf8f2;color:#29352e;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px;vertical-align:top}.scroll{overflow:auto}img{max-width:100%}td img{width:160px;max-width:none}details{background:white;margin:14px 0;padding:12px}a{color:#285b47}summary{cursor:pointer}small{display:block}</style>'


def run(root):
    root = Path(root); out = root/'analysis'
    if out.exists(): raise FileExistsError('No overwrite')
    out.mkdir(); (out/'images').mkdir()
    data = json.loads((root/'extraction/features.json').read_text())
    models = json.loads((root/'regression/results.json').read_text())
    with (root/'regression/predictions.csv').open() as f: predictions = list(csv.DictReader(f))
    with (root/'regression/feature_stability.csv').open() as f: feature_rows = list(csv.DictReader(f))
    predicted = {(r['image_id'],r['transform'],r['pipeline'],r['model']):r for r in predictions}
    pipelines = [c['name'] for c in data['configuration']]
    transforms = list(dict.fromkeys(r['transform'] for r in data['inputs']))
    names = ['linear','ridge','random_forest','gradient_boosting']
    fig, axs = plt.subplots(2, 2, figsize=(17, 11))
    for ax, name in zip(axs.flat, names):
        matrix = np.array([[next(s['max_absolute_change_g'] for s in models['sensitivity'] if s['pipeline']==p and s['model']==name and s['transform']==t) for t in transforms[1:]] for p in pipelines])
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=75)
        ax.set_yticks(range(len(pipelines)), pipelines);ax.set_xticks(range(len(transforms)-1), transforms[1:], rotation=70, ha='right');ax.set_title(name)
        for (i,j), v in np.ndenumerate(matrix): ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=7,color='black')
    fig.colorbar(im, ax=axs.ravel().tolist(), label='Maximum paired prediction change (g)', shrink=.6)
    fig.suptitle('Same-original derivatives ONLY | not independent weight accuracy')
    fig.subplots_adjust(left=.12,right=.87,bottom=.2,top=.93,hspace=.85,wspace=.4)
    fig.savefig(out/'prediction_stability.png',dpi=150);plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,7))
    markers = dict(linear='o',ridge='s',random_forest='^',gradient_boosting='D',mean='x')
    for p, color in zip(pipelines,plt.cm.tab10.colors):
        for fit in [f for f in models['fits'] if f['pipeline']==p]:
            ax.scatter(fit['training_metrics']['mae_g'],fit['max_standard_transform_change_g'],color=color,marker=markers[fit['model']],label=p if fit['model']=='ridge' else None,s=70)
            if fit['model']=='ridge':ax.annotate(p,(fit['training_metrics']['mae_g'],fit['max_standard_transform_change_g']),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Training MAE (g) — NOT generalization error');ax.set_ylabel('Maximum same-original change, excluding canvas scale (g)');ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'stability_vs_fit.png',dpi=150);plt.close(fig)
    summary=[]
    for p in pipelines:
        for key in data['feature_keys'][p]:
            values=[abs(float(r['delta_over_original_image_std'])) for r in feature_rows if r['pipeline']==p and r['feature']==key and r['transform']!='canvas_scale75']
            half=[abs(float(r['delta_over_original_image_std'])) for r in feature_rows if r['pipeline']==p and r['feature']==key and r['transform']=='resize50']
            summary.append(dict(pipeline=p,feature=key,mean_abs_delta_over_original_std=float(np.mean(values)),p95_abs_delta_over_original_std=float(np.percentile(values,95)),max_abs_delta_over_original_std=max(values),half_size_max_abs_delta_over_original_std=max(half)))
    csv_write(out/'feature_stability_summary.csv',summary)
    fig, axs = plt.subplots(2,3,figsize=(17,10))
    for ax,p in zip(axs.flat,pipelines):
        rs=[r for r in summary if r['pipeline']==p];ax.barh([r['feature'] for r in rs],[r['p95_abs_delta_over_original_std'] for r in rs],color='#5d7765');ax.set_title(p);ax.set_xlabel('P95 |delta| / std across 15 originals');ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Feature stability under image perturbations | denominator is descriptive, not population uncertainty');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'feature_stability.png',dpi=150);plt.close(fig)
    scalar_keys=sorted(set(k for r in data['rows'] for k,v in r.items() if not isinstance(v,(list,dict))))
    with (out/'all_features.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys,extrasaction='ignore');writer.writeheader();writer.writerows(data['rows'])
    csv_write(out/'morphology_ablation.csv',data['morphology_ablations'])
    ids=list(dict.fromkeys(r['image_id'] for r in data['inputs']))
    for ident in ids:
        page='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'+STYLE+f'<title>{ident} 安定性比較</title><a href="../index.html">比較一覧へ</a><h1>{ident}</h1><p>同じ原本の派生画像。すべて未承認の研究値です。独立した重量精度ではありません。紫線は候補マスクの輪郭。画像を押すとマスクPNGを開きます。</p>'
        for t in transforms:
            page+=f'<details {"open" if t in ("original","resize50") else ""}><summary>{t}</summary><div class="scroll"><table><tr><th>変換入力（表示用JPEG）</th>'+''.join(f'<th>{p}</th>' for p in pipelines)+'</tr><tr>'
            page+=f'<td><img src="../../extraction/{ident}/{t}/input_preview.jpg"></td>'
            for p in pipelines:
                row=next(r for r in data['rows'] if r['image_id']==ident and r['transform']==t and r['pipeline']==p)
                page+=f'<td><a href="../../extraction/{row["mask_path"]}"><img loading="lazy" src="../../extraction/{row["overlay_path"]}"></a><small>占有率 {row["mask_fraction"]:.4f}<br>自己IoU {row["self_mask_iou"] if row["self_mask_iou"] is None else round(row["self_mask_iou"],4)}<br>除去率 {row["removed_fraction"]:.4f}<br>QC {html.escape(str(row["qc_flags"]))}</small></td>'
            page+='</tr></table><table><tr><th>研究予測g（括弧内は元画像との差g）</th>'+''.join(f'<th>{p}</th>' for p in pipelines)+'</tr>'
            for name in names:
                page+=f'<tr><th>{name}</th>'
                for p in pipelines:
                    pred=predicted[(ident,t,p,name)];page+=f'<td>{float(pred["research_prediction_g"]):.2f} ({float(pred["delta_from_original_g"]):+.2f})</td>'
                page+='</tr>'
            page+='</table></div></details>'
        (out/'images'/f'{ident}.html').write_text(page+'</html>',encoding='utf-8')
    page='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'+STYLE+'<title>方式B：安定性研究</title><h1>方式B：安定性・再現性の比較</h1><p>15原本・5撮影群、物理的独立性は未確認。225入力は同じ15原本の派生です。独立評価MAE/RMSEは未算出。数値は研究用で、本番採用なし。</p><p><a href="../REPORT_JA.md">日本語報告</a> / <a href="all_features.csv">全1350行の特徴量</a> / <a href="../regression/predictions.csv">全6750行の予測</a> / <a href="feature_stability_summary.csv">特徴安定性</a> / <a href="morphology_ablation.csv">形態学処理の比較</a></p><h2>画像・マスク・予測を確認</h2><p>'
    page+=' / '.join(f'<a href="images/{i}.html">{i}</a>' for i in ids)+'</p><h2>訓練内誤差と変動</h2><div class="scroll"><table><tr><th>Pipeline</th><th>Model</th><th>訓練MAE g</th><th>訓練RMSE g</th><th>最大変動 g（canvas除外）</th></tr>'
    for fit in models['fits']:
        page+=f'<tr><td>{fit["pipeline"]}</td><td>{fit["model"]}</td><td>{fit["training_metrics"]["mae_g"]:.2f}</td><td>{fit["training_metrics"]["rmse_g"]:.2f}</td><td>{fit["max_standard_transform_change_g"]:.2f}</td></tr>'
    page+='</table></div>'+''.join(f'<h2>{title}</h2><a href="{file}"><img src="{file}"></a>' for file,title in [('prediction_stability.png','変換別・最大予測変動'),('feature_stability.png','特徴量の安定性'),('stability_vs_fit.png','安定性と訓練内当てはまり')])
    (out/'index.html').write_text(page+'</html>',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);run(parser.parse_args().root)
