"""Shared distinction between missing dry-pasta nutrition and valid zero nutrients."""
import math

PRODUCT_NUTRIENTS = ('kcal_100', 'protein_100', 'fat_100', 'carb_100')

# USDA National Nutrient Database for Standard Reference, Release 28.
# NDB 20420, Pasta, dry, unenriched, per 100 g.
STANDARD_DRY_PASTA = {
    'id': None, 'name': '一般的な乾麺（概算）', 'g_per_strand': None,
    'kcal_100': 371.0, 'protein_100': 13.04, 'fat_100': 1.51, 'carb_100': 74.67,
}
STANDARD_NUTRITION_NOTE = '一般的な乾麺の概算（米国農務省 USDA SR28、食品番号20420）'


def standard_nutrition_used(product):
    return (product.get('name', product.get('product_name')) == STANDARD_DRY_PASTA['name']
            and product.get('g_per_strand') is None)


def has_nutrition(product, *, allow_demo=False):
    if not allow_demo and str(product.get('name', product.get('product_name', ''))).startswith('デモ用'):
        return False
    try:
        values = [float(product[key]) for key in PRODUCT_NUTRIENTS]
    except (KeyError, TypeError, ValueError):
        return False
    # The existing products schema defaults missing fields to zero. For dry pasta,
    # zero calories cannot establish that a nutritional label has been provided.
    return values[0] > 0 and all(math.isfinite(value) and value >= 0 for value in values)


def snapshot_for_nutrition(product, *, allow_demo=False):
    snapshot = dict(product)
    if not has_nutrition(product, allow_demo=allow_demo):
        snapshot.update(dict.fromkeys(PRODUCT_NUTRIENTS))
    return snapshot
