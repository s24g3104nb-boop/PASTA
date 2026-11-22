"""Frozen OpenCLIP prompts for an editable cooking suggestion, offline only."""
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL = 'ViT-B-32-quickgelu'
CHECKPOINT_SHA256 = '46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa'
DISH_PROMPTS = {
    'トマトパスタ': ('cooked spaghetti with red tomato sauce', 'a plated pasta dish with bright red marinara sauce'),
    'ジェノベーゼ': ('cooked spaghetti with green basil pesto', 'a plated green pesto pasta dish'),
    'カルボナーラ': ('cooked spaghetti carbonara with egg and cheese', 'a plated pale yellow spaghetti carbonara with bacon'),
    'その他': ('a plate of creamy Alfredo pasta', 'a photograph of uncooked dry spaghetti', 'a photo of another meal or a nonfood object'),
}
INGREDIENT_PROMPTS = {
    'ベーコン片': ('a plate of pasta with clearly visible pieces of bacon', 'a plate of pasta without visible bacon pieces'),
    'トマトの果肉': ('cooked pasta with visible red tomato pieces', 'cooked pasta without visible pieces of tomato'),
    '葉のハーブ': ('cooked pasta with visible green herb leaves on top', 'cooked pasta without visible herb leaves'),
    '削ったチーズ': ('pasta with clearly visible shaved cheese on top', 'pasta without visible pieces of shaved cheese'),
}
CANDIDATE_MIN_SIMILARITY = .20
CANDIDATE_MIN_MARGIN = .012
VISIBLE_HINT_MIN_MARGIN = .026


@dataclass(frozen=True)
class Suggestion:
    name: str | None
    visible_options: tuple[str, ...]
    color_note: str | None
    method: str


def color_note(rgb):
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb[h//5:h-h//5, w//5:w-w//5], cv2.COLOR_RGB2HSV)
    hue, sat, val = np.moveaxis(hsv, -1, 0)
    red = ((hue < 9) | (hue >= 167)) & (sat > 100) & (val > 45)
    green = (hue >= 32) & (hue <= 90) & (sat > 55) & (val > 45)
    cream = (hue >= 12) & (hue < 40) & (sat >= 20) & (sat < 100) & (val > 110)
    colors = ('赤', '緑', '黄・クリーム')
    fractions = [float(pixels.mean()) for pixels in (red, green, cream)]
    order = np.argsort(fractions)
    if fractions[order[-1]] >= .12 and fractions[order[-1]] - fractions[order[-2]] >= .07:
        return colors[order[-1]]
    return None


def suggest_from_scores(dishes, ingredients, color):
    ranked = sorted(dishes, key=dishes.get, reverse=True)
    top, runner_up = ranked[:2]
    # Similarity and margin are provisional decision rules, not probabilities.
    name = top if (top != 'その他' and dishes[top] >= CANDIDATE_MIN_SIMILARITY
                   and dishes[top] - dishes[runner_up] >= CANDIDATE_MIN_MARGIN) else None
    hints = tuple(key for key, (positive, negative) in ingredients.items()
                  if positive >= CANDIDATE_MIN_SIMILARITY and positive - negative >= VISIBLE_HINT_MIN_MARGIN) if name else ()
    return Suggestion(name, hints, color, 'OpenCLIP ViT-B/32 LAION400M')


class LocalDishClassifier:
    def __init__(self, checkpoint, device='cpu'):
        path = Path(checkpoint).resolve(strict=True)
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != CHECKPOINT_SHA256:
                raise ValueError('Unexpected OpenCLIP checkpoint')
        import open_clip
        import torch

        if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Local device is unavailable')
        torch.set_num_threads(1)
        self.torch, self.device = torch, device
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=str(path), device=device, weights_only=True)
        self.model = model.eval().requires_grad_(False)
        prompts = [text for group in (*DISH_PROMPTS.values(), *INGREDIENT_PROMPTS.values()) for text in group]
        with torch.inference_mode():
            tokens = open_clip.get_tokenizer(MODEL)(prompts).to(device)
            self.text = self.model.encode_text(tokens, normalize=True)

    def predict(self, rgb):
        with self.torch.inference_mode():
            pixels = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            visual = self.model.encode_image(pixels, normalize=True)
            scores = (visual @ self.text.T)[0].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Invalid image classifier result')
        cursor = 0
        dishes = {}
        for label, prompts in DISH_PROMPTS.items():
            group = scores[cursor:cursor+len(prompts)]
            dishes[label] = float(group.max() if label == 'その他' else group.mean())
            cursor += len(prompts)
        ingredients = {}
        for label in INGREDIENT_PROMPTS:
            ingredients[label] = tuple(scores[cursor:cursor+2])
            cursor += 2
        return suggest_from_scores(dishes, ingredients, color_note(rgb))
