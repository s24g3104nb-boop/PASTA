"""Local colour and optional OpenCLIP suggestions for the standard recipes."""
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__:
    from .dish_classifier import LinearDishHead, STANDARD_DISH_NAMES
else:
    from dish_classifier import LinearDishHead, STANDARD_DISH_NAMES

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
COLOR_METHOD = 'OpenCV'


@dataclass(frozen=True)
class Suggestion:
    name: str | None
    visible_options: tuple[str, ...]
    color_note: str | None
    method: str


def _balance_from_neutral_pixels(rgb, outside):
    # A mostly neutral plate can indicate a mild lighting cast. Colored plates
    # and dark backgrounds provide no reliable white reference.
    values = rgb[outside].astype(np.float32)
    maximum = values.max(axis=1)
    neutral = (maximum >= 115) & (values.min(axis=1) >= .79 * maximum)
    if np.count_nonzero(neutral) < .08 * len(values):
        return rgb
    reference = np.median(values[neutral], axis=0)
    gains = np.clip(reference.mean() / reference, .84, 1.18)
    return np.rint(np.clip(rgb * gains, 0, 255)).astype(np.uint8)


def _sauce_colors(rgb):
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    lightness, red_green, blue_yellow = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    # Lab a* helps separate orange-red sauce from brown wood and yellow-green
    # pesto from warm cream. OpenCV centers a* and b* at 128.
    tomato = (((hue <= 16) | (hue >= 166)) & (saturation >= 84) & (value >= 55)
              & (red_green >= 150))
    # Pale yellow can acquire a slightly green Lab a* under cool light or
    # when red and green channels approach clipping.
    bright_yellow = (value >= 230) & (hue <= 36) & (saturation <= 112)
    pale_yellow = ((value >= 170) & (hue <= 39) & (saturation <= 83)
                   & (red_green >= 116)) | bright_yellow
    genovese = ((hue >= 23) & (hue <= 86) & (saturation >= 40) & (value >= 47)
                 & (red_green <= 125) & (blue_yellow >= 135) & ~pale_yellow)
    # A golden egg sauce can be more saturated than pale cream. Require a
    # yellower hue for those pixels to avoid red-orange toppings and wood.
    yellow_or_cream = ((hue >= 12) & (saturation <= 136)) | ((hue >= 16) & (saturation <= 205))
    carbonara = (yellow_or_cream & (hue <= 39) & (saturation >= 22)
                 & (value >= 95) & (lightness >= 82) & (blue_yellow >= 140)
                 & ((red_green >= 126) | pale_yellow) & (red_green <= 148))
    return {'トマトパスタ': ('赤・オレンジ', tomato), 'ジェノベーゼ': ('緑・黄緑', genovese),
            'カルボナーラ': ('黄・クリーム', carbonara)}


def suggest_dish_from_color(rgb):
    """Suggest a recipe from central sauce colour and spatial support only."""
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8 or min(rgb.shape[:2]) == 0):
        raise ValueError('Expected a nonempty RGB uint8 photo')
    h, w = rgb.shape[:2]
    scale = min(1., 384 / max(h, w))
    shape = max(16, round(w * scale)), max(16, round(h * scale))
    if shape != (w, h):
        rgb = cv2.resize(rgb, shape, interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
    y = (2. * (np.arange(h) + .5) / h - 1.)[:, None]
    x = (2. * (np.arange(w) + .5) / w - 1.)[None, :]
    radius_squared = x*x + y*y
    middle = radius_squared <= .28
    outside = radius_squared >= .48
    weights = np.maximum(0., 1. - radius_squared)**2
    # A single basil leaf or piece of bacon can be intensely coloured without
    # being typical of the sauce. Check distribution across the central dish.
    center = radius_squared <= .49
    quadrants = [center & (x * dx > 0) & (y * dy > 0) for dx in (-1, 1) for dy in (-1, 1)]
    colors = _sauce_colors(_balance_from_neutral_pixels(rgb, outside))
    scores = {}
    for name, (_, mask) in colors.items():
        inner = float(mask[middle].mean())
        outer = float(mask[outside].mean())
        support = .65 * inner + .35 * float(np.average(mask, weights=weights))
        spread = sum(min(float(mask[q].mean()) / .12, 1.) for q in quadrants) / len(quadrants)
        scores[name] = max(0., support * (.70 + .30 * spread) - .70 * outer)
    # This is a choice among the three recipes. In an evidence-free tie, use
    # the same ordering as the manual menu.
    name = max(STANDARD_DISH_NAMES, key=scores.get)
    note = colors[name][0] if scores[name] > 0 else None
    return Suggestion(name, (), note, COLOR_METHOD)


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


class OpenClipVision:
    """Shared local image encoder with pinned official checkpoint."""

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
    def encode(self, image):
        with self.torch.inference_mode():
            pixels = self.preprocess(image).unsqueeze(0).to(self.device)
            return self.model.encode_image(pixels, normalize=True)


class LocalDishClassifier(OpenClipVision):
    def __init__(self, checkpoint, device='cpu', *, head_path=None):
        super().__init__(checkpoint, device)
        self.head = LinearDishHead.load(head_path, CHECKPOINT_SHA256) if head_path else None
        import open_clip

        prompts = [text for group in (*DISH_PROMPTS.values(), *INGREDIENT_PROMPTS.values()) for text in group]
        with self.torch.inference_mode():
            tokens = open_clip.get_tokenizer(MODEL)(prompts).to(device)
            self.text = self.model.encode_text(tokens, normalize=True)

    def image_evidence(self, rgb):
        with self.torch.inference_mode():
            visual = self.encode(Image.fromarray(rgb))
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
        return visual[0].float().cpu().numpy(), suggest_from_scores(dishes, ingredients, color_note(rgb))

    def predict(self, rgb):
        embedding, suggestion = self.image_evidence(rgb)
        if self.head is None:
            return suggestion
        name = self.head.select(embedding, suggestion.name)
        return Suggestion(name, suggestion.visible_options if name else (), suggestion.color_note,
                          'OpenCLIP ViT-B/32 + supervised linear head')
