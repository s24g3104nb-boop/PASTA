"""Standard-recipe calculation for a meal, independent of dry-pasta capture."""
from dataclasses import asdict

if __package__:
    from .cooking import Ingredient, NUTRIENT_KEYS, calculate_meal, dry_snapshot, nonnegative, template_ingredients
    from .nutrition_state import PRODUCT_NUTRIENTS, STANDARD_DRY_PASTA, STANDARD_NUTRITION_NOTE
    from .standard_recipes import recipe_choices
else:
    from cooking import Ingredient, NUTRIENT_KEYS, calculate_meal, dry_snapshot, nonnegative, template_ingredients
    from nutrition_state import PRODUCT_NUTRIENTS, STANDARD_DRY_PASTA, STANDARD_NUTRITION_NOTE
    from standard_recipes import recipe_choices


REFERENCE_DRY_G = 80.0


def _template(dish):
    template = next(iter(recipe_choices(dish).values()), None)
    if template is None:
        raise ValueError('標準レシピの料理を選んでください。')
    return template


def standard_rows(dish, weight):
    return [asdict(row) for row in template_ingredients(_template(dish), weight)]


def recipe_dry_basis(recipe, measurement=None):
    weight_source = 'reference'
    weight_value = recipe.get('reference_dry_g')
    if measurement is not None:
        has_nutrition = (all(measurement.get(key) is not None for key in PRODUCT_NUTRIENTS)
                         or all(measurement.get(key) is not None for key in NUTRIENT_KEYS))
        if has_nutrition:
            return dry_snapshot(measurement)
        # Older or manual measurements may have a usable weight without a nutrition snapshot.
        weight_source = 'estimated' if measurement.get('actual_g') is None else 'actual'
        weight_value = measurement.get(weight_source + '_g')
    weight = nonnegative(weight_value, '乾麺の重量')
    if weight <= 0:
        raise ValueError('乾麺の重量は0gより大きくしてください。')
    return dict(dry_weight_g=weight, weight_source=weight_source,
                nutrition={nutrient: STANDARD_DRY_PASTA[key] * weight / 100
                           for nutrient, key in zip(NUTRIENT_KEYS, PRODUCT_NUTRIENTS)},
                nutrition_source=STANDARD_NUTRITION_NOTE)


def calculate_finished_recipe(recipe, measurement=None):
    """Recompute from the selected dry record or reference amount and entered materials."""
    dry = recipe_dry_basis(recipe, measurement)
    dish = recipe.get('dish')
    weight = dry['dry_weight_g']
    ingredients = [Ingredient(**row).validated() for row in recipe['ingredients']]
    defaults = standard_rows(dish, weight)
    confirmed = recipe.get('ingredients_confirmed') is True
    basis = 'confirmed_amounts' if confirmed and dry['weight_source'] == 'actual' else 'template_estimate'
    percent = nonnegative(recipe.get('consumed_percent'), '食べる割合')
    result = calculate_meal(dry, ingredients, consumed_fraction=percent / 100,
                            nutrition_basis=basis)
    if basis == 'template_estimate':
        result['assumptions'] = '乾麺の重量と入力した材料に基づく概算。標準レシピの分量には参考値を含みます。'
    result['recipe_input'] = dict(dish=dish, reference_dry_g=weight if measurement is None else None,
                                  ingredients=result['ingredients'], ingredients_confirmed=confirmed,
                                  consumed_percent=percent)
    result['dry_weight_g'] = weight
    result['dry_weight_source'] = dry['weight_source']
    result['dry_nutrition_source'] = dry['nutrition_source']
    result['ingredient_origins'] = ['standard_recipe' if row in defaults else 'user_input'
                                    for row in result['ingredients']]
    result['recipe_source'] = _template(dish)['source']
    return result
