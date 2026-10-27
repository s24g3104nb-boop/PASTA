"""乾麺の履歴と利用者が確認した材料から料理の栄養を計算する。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


NUTRIENT_KEYS = ('kcal', 'protein_g', 'fat_g', 'carb_g')


def nonnegative(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{label}には数値を入力してください。')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label}が未入力、または数値ではありません。') from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{label}には有限の0以上の数値を入力してください。')
    return number


@dataclass(frozen=True)
class Ingredient:
    name: str
    amount_g: float
    kcal_100: float
    protein_100: float
    fat_100: float
    carb_100: float

    def validated(self) -> 'Ingredient':
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('材料名を入力してください。')
        return Ingredient(self.name.strip(), **{
            key: nonnegative(getattr(self, key), f'{self.name}: {key}')
            for key in ('amount_g', 'kcal_100', 'protein_100', 'fat_100', 'carb_100')
        })

    def nutrition(self) -> dict[str, float]:
        item = self.validated()
        scale = item.amount_g / 100
        return dict(zip(NUTRIENT_KEYS, (item.kcal_100 * scale, item.protein_100 * scale,
                                       item.fat_100 * scale, item.carb_100 * scale)))


def dry_snapshot(record: Mapping) -> dict:
    """保存済みの栄養値を固定し、実測がある場合は重量比で補正する。"""
    estimated = nonnegative(record.get('estimated_g'), '保存された推定重量')
    if estimated <= 0:
        raise ValueError('推定重量が0の履歴から栄養値を復元できません。')
    actual = record.get('actual_g')
    weight = estimated if actual is None else nonnegative(actual, '保存された実測重量')
    if weight <= 0:
        raise ValueError('料理に使用した乾麺重量は0gより大きい必要があります。')
    product_keys = ('kcal_100', 'protein_100', 'fat_100', 'carb_100')
    if all(record.get(key) is not None for key in product_keys):
        nutrition = {key: nonnegative(record[source], f'乾麺の{source}') * weight / 100
                     for key, source in zip(NUTRIENT_KEYS, product_keys)}
        nutrition_source = '計測時に保存した製品の100g栄養値'
    else:
        nutrition = {key: nonnegative(record.get(key), f'乾麺の{key}') * weight / estimated
                     for key in NUTRIENT_KEYS}
        nutrition_source = '旧履歴の保存済み乾麺栄養値を重量比で補正（丸め誤差あり）'
    if not all(math.isfinite(v) for v in nutrition.values()):
        raise ValueError('乾麺の栄養計算が数値の範囲を超えました。')
    return {'measurement_id': record['id'], 'product_name': record['product_name'],
            'dry_weight_g': weight, 'weight_source': 'actual' if actual is not None else 'estimated',
            'nutrition': nutrition, 'source_kind': record.get('source_kind', 'unknown'),
            'nutrition_source': nutrition_source}


def calculate_meal(snapshot: Mapping, ingredients: Sequence[Ingredient], *,
                   servings: int | None = None, consumed_fraction: float | None = None) -> dict:
    if (servings is None) == (consumed_fraction is None):
        raise ValueError('等分人数か食べる割合のどちらか一方を指定してください。')
    if servings is not None:
        if isinstance(servings, bool) or not isinstance(servings, int) or servings < 1:
            raise ValueError('人数は1以上の整数で指定してください。')
        fraction = 1 / servings
    else:
        fraction = nonnegative(consumed_fraction, '食べる割合')
        if fraction <= 0 or fraction > 1:
            raise ValueError('食べる割合は0より大きく1以下にしてください。')
    dry = {key: nonnegative(snapshot['nutrition'].get(key), f'乾麺の{key}')
           for key in NUTRIENT_KEYS}
    validated = [item.validated() for item in ingredients]
    try:
        additional = {key: math.fsum(item.nutrition()[key] for item in validated)
                      for key in NUTRIENT_KEYS}
    except OverflowError:
        raise ValueError('材料の栄養計算が数値の範囲を超えました。') from None
    total = {key: dry[key] + additional[key] for key in NUTRIENT_KEYS}
    if not all(math.isfinite(v) for v in total.values()):
        raise ValueError('栄養計算が数値の範囲を超えました。')
    return {'dry': dry, 'additional': additional, 'total': total,
            'portion': {key: value * fraction for key, value in total.items()},
            'servings': servings, 'consumed_fraction': fraction,
            'ingredients': [asdict(item) for item in validated],
            'assumptions': '利用者が入力・確認した全材料と使用量に基づく計算。写真から材料量は分かりません。'}
