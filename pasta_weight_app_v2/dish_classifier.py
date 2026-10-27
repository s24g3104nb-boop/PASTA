"""料理名分類の接続口。現在はモデル未学習のため常に判別不能を返す。"""
from dataclasses import dataclass


DISH_NAMES = ('カルボナーラ', 'ボロネーゼ', 'ミートソース', 'その他／不明')
INGREDIENT_HINTS = {
    'カルボナーラ': ('卵', 'チーズ', 'ベーコン', '油', '調味料'),
    'ボロネーゼ': ('ひき肉', 'トマト／ソース', '野菜', '油', 'チーズ', '調味料'),
    'ミートソース': ('ひき肉', 'トマト／ソース', '野菜', '油', '調味料'),
}


@dataclass(frozen=True)
class DishPrediction:
    label: str | None
    score: float | None
    status: str
    message: str


class UntrainedDishClassifier:
    def predict(self, image=None) -> DishPrediction:
        return DishPrediction(None, None, 'untrained',
                              '料理分類モデルは未学習です。料理名を手動で選択してください。')
