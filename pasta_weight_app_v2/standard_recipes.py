"""Editable serving examples. Nutrients per 100 g from USDA SR28."""

# NDB codes 11549, 02044, 01033, 04053, 01123, 10860, 12147 and 11215.
FOODS = {
    'トマトソース（缶詰）': (24, 1.2, .3, 5.31),
    'バジル（生）': (23, 3.15, .64, 2.65),
    'パルメザン（硬質）': (392, 35.75, 25.83, 3.22),
    'オリーブ油': (884, 0, 100, 0),
    '全卵（生・殻を除く）': (143, 12.56, 9.51, .72),
    'ベーコン（焼き上がり）': (548, 35.73, 43.27, 1.35),
    '松の実（乾燥）': (673, 13.69, 68.37, 13.08),
    'にんにく（生）': (149, 6.36, .5, 33.06),
}
EXAMPLES = {
    -1: ('トマトパスタ', [('トマトソース（缶詰）', 100), ('オリーブ油', 8), ('にんにく（生）', 3)]),
    -2: ('ジェノベーゼ', [('バジル（生）', 15), ('オリーブ油', 15), ('松の実（乾燥）', 8),
                         ('パルメザン（硬質）', 12), ('にんにく（生）', 3)]),
    -3: ('カルボナーラ', [('全卵（生・殻を除く）', 50), ('ベーコン（焼き上がり）', 25),
                          ('パルメザン（硬質）', 15)]),
}
FIELDS = ('kcal_100', 'protein_100', 'fat_100', 'carb_100')


def standard_recipe(ident):
    match = EXAMPLES.get(ident)
    if match is None:
        return None
    dish, amounts = match
    return dict(id=ident, selected_dish=dish, name=dish+'の標準レシピ（概算）',
                reference_dry_g=100, default_servings=1,
                source='USDA National Nutrient Database SR28（100g当たり）。分量は乾麺100gに対する参考例。',
                ingredients=[dict(name=name, amount_g=amount, **dict(zip(FIELDS, FOODS[name])))
                             for name, amount in amounts])


def recipe_choices(dish):
    return {ident: standard_recipe(ident) for ident, (label, _) in EXAMPLES.items() if label == dish}
