from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import (grouping, image_rgb, import_dataset, load_dataset,
                      split_dataset, validate_splits, write_csv)
from .evaluate import evaluate, load_calibrations
from .model import load_model, predict, train


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pasta Lens 乾麺データ・実験用回帰。実物精度は未検証。')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'import', 'split', 'train', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('csv', help='ラベルCSV（UTF-8）')
        p.add_argument('--image-root', help='画像相対パスの基点。既定: CSVのディレクトリ')
        if command == 'import':
            p.add_argument('--destination', required=True)
        if command == 'split':
            p.add_argument('--output', required=True)
            p.add_argument('--seed', type=int, default=42)
        if command in ('train', 'evaluate'):
            p.add_argument('--output', required=True)
            p.add_argument('--allow-synthetic', action='store_true', help='合成画像のみの実装テスト用')
        if command == 'train':
            p.add_argument('--alpha', type=float, default=1.)
        if command == 'evaluate':
            choice = p.add_mutually_exclusive_group(required=True)
            choice.add_argument('--model')
            choice.add_argument('--calibrations', help='OpenCV用: product_id,g_per_strandのCSV')
            p.add_argument('--split', choices=('val', 'test'), default='val')
            p.add_argument('--final-test', action='store_true', help='最終評価でtestを開封することを明示')
    p = sub.add_parser('predict')
    p.add_argument('image')
    p.add_argument('--model', required=True)
    p.add_argument('--mode', required=True, choices=('spread', 'bundle_end', 'bundle'))
    p.add_argument('--product-id', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'predict':
            result = predict(load_model(args.model), image_rgb(args.image), args.mode, args.product_id)
        else:
            samples = load_dataset(args.csv, args.image_root)
            if args.command == 'validate':
                if any(s.values.get('split') for s in samples):
                    validate_splits(samples)
                result = dict(images=len(samples), independent_groups=len(grouping(samples)),
                              modes=dict(Counter(s.values['mode'] for s in samples)),
                              sources=dict(Counter(s.values['source'] for s in samples)),
                              weights_g=sorted({s.weight for s in samples}),
                              warning='近似画像は保守的に同一グループ化します。分割後も人手で類似写真・条件偏りを確認してください。')
            elif args.command == 'import':
                result = dict(dataset=str(import_dataset(samples, args.destination)))
            elif args.command == 'split':
                split_dataset(samples, args.seed)
                write_csv(samples, args.output, args.image_root)
                result = dict(output=args.output, splits=dict(Counter(s.values['split'] for s in samples)))
            elif args.command == 'train':
                model = train(samples, args.output, args.alpha, args.allow_synthetic)
                result = dict(model=args.output, source=model['source'], validation=model['validation_report']['overall'])
            else:
                if args.split == 'test' and not args.final_test:
                    raise ValueError('testは最終評価専用です。調整はvalで行い、最後に --final-test を明示してください')
                marker = Path(str(args.model) + '.test-used.json') if args.model else None
                if args.split == 'test' and marker is not None and marker.exists():
                    raise ValueError('このモデルのtestは評価済みです。保存済みレポートを使用してください')
                result = evaluate(samples, args.split,
                                  model=load_model(args.model) if args.model else None,
                                  calibrations=load_calibrations(args.calibrations) if args.calibrations else None,
                                  allow_synthetic=args.allow_synthetic)
                save_json(result, args.output)
                if args.split == 'test' and marker is not None:
                    save_json(dict(report=str(Path(args.output).resolve()),
                                   dataset_sha256=result['dataset_sha256']), marker)
                result = dict(report=args.output, overall=result['overall'], source=result['source'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'エラー: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
