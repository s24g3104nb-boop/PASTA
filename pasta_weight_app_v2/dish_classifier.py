"""Cooked dish recognition labels and a local, optional linear classifier."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


STANDARD_DISH_NAMES = ('カルボナーラ', 'トマトパスタ', 'ジェノベーゼ')
DISH_NAMES = (*STANDARD_DISH_NAMES, 'ボロネーゼ', 'ミートソース', 'その他／不明')
INGREDIENT_HINTS = {
    'カルボナーラ': ('卵', 'チーズ', 'ベーコン', '油', '調味料'),
    'トマトパスタ': ('トマトソース', 'にんにく', 'オリーブ油'),
    'ジェノベーゼ': ('バジル', 'オリーブ油', '松の実', 'チーズ'),
    'ボロネーゼ': ('ひき肉', 'トマト／ソース', '野菜', '油', 'チーズ', '調味料'),
    'ミートソース': ('ひき肉', 'トマト／ソース', '野菜', '油', 'チーズ', '調味料'),
}
EMBEDDING_SIZE = 512
MIN_CLASS_SCORE = 0.60
MIN_CLASS_MARGIN = 0.18


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


@dataclass(frozen=True)
class LinearDishHead:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def load(cls, path, checkpoint_sha256):
        path = Path(path)
        if path.stat().st_size > 256_000:
            raise ValueError('Dish classifier artifact is too large')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1:
            raise ValueError('Unrecognized dish classifier version')
        if data.get('labels') != list(STANDARD_DISH_NAMES) or data.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('Dish classifier does not match the image encoder')
        try:
            weights = np.asarray(data.get('weights'), dtype=float)
            bias = np.asarray(data.get('bias'), dtype=float)
        except (ValueError, TypeError, OverflowError):
            raise ValueError('Invalid dish classifier coefficients') from None
        if weights.shape != (len(STANDARD_DISH_NAMES), EMBEDDING_SIZE) or bias.shape != (len(STANDARD_DISH_NAMES),):
            raise ValueError('Unexpected dish classifier dimensions')
        if not (np.isfinite(weights).all() and np.isfinite(bias).all()):
            raise ValueError('Invalid dish classifier coefficients')
        return cls(weights, bias)

    def artifact(self, checkpoint_sha256):
        return dict(version=1, labels=list(STANDARD_DISH_NAMES), checkpoint_sha256=checkpoint_sha256,
                    weights=self.weights.tolist(), bias=self.bias.tolist())

    def select(self, embedding, reference_label):
        # A head only trained on the three recipes cannot recognize an arbitrary out-of-scope dish.
        # Use the encoder's prompt-based abstention as a separate scope check.
        if reference_label not in STANDARD_DISH_NAMES:
            return None
        values = self.weights @ embedding + self.bias
        if not np.isfinite(values).all():
            return None
        scores = np.exp(values - values.max())
        scores /= scores.sum()
        first, second = np.argsort(scores)[-2:][::-1]
        if scores[first] < MIN_CLASS_SCORE or scores[first] - scores[second] < MIN_CLASS_MARGIN:
            return None
        return STANDARD_DISH_NAMES[first]
