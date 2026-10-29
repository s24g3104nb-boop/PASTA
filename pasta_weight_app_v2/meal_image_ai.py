"""Future cooked-meal inference boundary. No trained implementation is shipped."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol
from meal_photos import PreparedMealPhoto


@dataclass(frozen=True)
class MealImagePrediction:
    status: Literal['unavailable', 'unsupported', 'rejected', 'estimated']
    model_version: str | None
    nutrition: Mapping[str, float] | None
    quality: Literal['unverified', 'rejected', 'validated']
    explanation: str


class MealImageEstimator(Protocol):
    """Future independent validation must precede connection to the diary."""
    def predict(self, photo: PreparedMealPhoto) -> MealImagePrediction: ...


class UnavailableMealImageEstimator:
    def predict(self, photo: PreparedMealPhoto) -> MealImagePrediction:
        return MealImagePrediction(
            status='unavailable', model_version=None, nutrition=None,
            quality='unverified', explanation='写真AIによるカロリー推定は準備中です。',
        )
