"""Shared distinction between missing dry-pasta nutrition and valid zero nutrients."""
import math

PRODUCT_NUTRIENTS = ('kcal_100', 'protein_100', 'fat_100', 'carb_100')


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
