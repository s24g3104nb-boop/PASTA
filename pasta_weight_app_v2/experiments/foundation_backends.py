"""Optional local adapters. No remote loading or production dependencies."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from .stability_pipeline import PipelineConfig, clean_mask, raw_mask

MAX_SIDE = 960
TEXT_PROMPT = 'dry uncooked spaghetti'
CONCEPT_THRESHOLD = 0.5
SEED = PipelineConfig('foundation_seed', MAX_SIDE, 'hsv_lab', True, 'object4')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mask_union(masks, shape):
    """Union visible instances. The result carries no strand count."""
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros(shape, dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != shape or not np.isin(masks, [0, 1]).all():
        raise ValueError('Expected source-aligned binary model masks')
    return masks.any(axis=0)


def hsv_mask(rgb, *, lab=False):
    if lab:
        return clean_mask(raw_mask(rgb, SEED), SEED) != 0
    return raw_mask(rgb, PipelineConfig('foundation_hsv', MAX_SIDE, 'hsv', False, 'object4',
                                      legacy_saturation_min=20)) != 0


def seed_boxes(rgb):
    """Use HSV+Lab proposals so SAM 2 receives the same prompts for every run."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(hsv_mask(rgb, lab=True).astype(np.uint8))
    height, width = rgb.shape[:2]
    significant = [box for box in stats[1:count] if box[4] >= max(12, height * width * .0002)]
    # Deterministic budget. No filenames, annotations, or measured weights here.
    boxes = sorted(significant, key=lambda box: (-int(box[4]), int(box[1]), int(box[0])))[:16]
    margin = max(4, round(max(height, width)*.015))
    return [np.array([max(0, x-margin), max(0, y-margin),
                      min(width-1, x+w-1+margin), min(height-1, y+h-1+margin)], np.float32)
            for x, y, w, h, _ in boxes]


class Opencv:
    def __init__(self, method):
        self.method = method
        self.metadata = dict(method=method, opencv_version=cv2.__version__,
                             mask_config_sha256=SEED.digest if method == 'opencv_lab' else None)

    def __call__(self, rgb):
        return hsv_mask(rgb, lab=self.method == 'opencv_lab')


class Sam2:
    def __init__(self, checkpoint, device, torch):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                                                     str(checkpoint), device=device,
                                                     apply_postprocessing=False))
        self.metadata = dict(method='sam2', config='sam2.1_hiera_tiny', prompts='hsv_lab_boxes_v1',
                             checkpoint_sha256=sha256_file(checkpoint), mask_config_sha256=SEED.digest,
                             sam2_version=version('SAM-2'))

    def __call__(self, rgb):
        return prompted_mask(self.predictor, self.predictor.reset_predictor, self.torch, rgb)


def prompted_mask(predictor, reset, torch, rgb):
    boxes = seed_boxes(rgb)
    masks = []
    if boxes:
        with torch.inference_mode():
            predictor.set_image(rgb)
            try:
                for box in boxes:
                    candidates, scores, _ = predictor.predict(box=box, multimask_output=False)
                    if len(scores):
                        masks.append(candidates[int(np.argmax(scores))])
            finally:
                reset()
    return mask_union(masks, rgb.shape[:2])


class MobileSam:
    def __init__(self, checkpoint, device, torch):
        from mobile_sam import SamPredictor, sam_model_registry

        self.torch = torch
        model = sam_model_registry['vit_t']()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.predictor = SamPredictor(model.to(device).eval().requires_grad_(False))
        self.metadata = dict(method='mobile_sam', config='vit_t', prompts='hsv_lab_boxes_v1',
                             checkpoint_sha256=sha256_file(checkpoint), mask_config_sha256=SEED.digest,
                             mobile_sam_version=version('mobile-sam'))

    def __call__(self, rgb):
        return prompted_mask(self.predictor, self.predictor.reset_image, self.torch, rgb)


class Sam31:
    def __init__(self, checkpoint, torch):
        from sam3.model_builder import build_sam3_predictor

        self.torch = torch
        self.predictor = build_sam3_predictor(checkpoint_path=str(checkpoint), version='sam3.1',
                                              compile=False, use_fa3=False, async_loading_frames=False)
        self.metadata = dict(method='sam31', checkpoint_sha256=sha256_file(checkpoint),
                             sam3_version=version('sam3'), prompt=TEXT_PROMPT,
                             output_prob_threshold=CONCEPT_THRESHOLD, mode='single_frame_multiplex')

    def __call__(self, rgb):
        with TemporaryDirectory(prefix='pasta-sam31-') as directory, self.torch.inference_mode():
            # Meta's multiplex runner reads a frame directory. Its local input is
            # freshly encoded and has no original path, EXIF, or label metadata.
            Image.fromarray(rgb).save(Path(directory)/'00000.jpg', quality=98)
            handler = self.predictor.handle_request
            session = handler(dict(type='start_session', resource_path=directory))['session_id']
            try:
                result = handler(dict(type='add_prompt', session_id=session, frame_index=0,
                                      text=TEXT_PROMPT, output_prob_thresh=CONCEPT_THRESHOLD))
                if result['frame_index'] != 0:
                    raise ValueError('Expected a result from the only input frame')
                return mask_union(result['outputs']['out_binary_masks'], rgb.shape[:2])
            finally:
                handler(dict(type='close_session', session_id=session))


def start_segmenter(method, checkpoint, device, torch):
    if method.startswith('opencv'):
        if checkpoint is not None or device != 'cpu':
            raise ValueError('OpenCV needs only CPU and no checkpoint')
        return Opencv(method)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise ValueError('Supply the official checkpoint as an existing local file')
    if method == 'mobile_sam':
        return MobileSam(checkpoint, device, torch)
    if method == 'sam2':
        return Sam2(checkpoint, device, torch)
    if device != 'cuda':
        raise ValueError('Meta SAM 3.1 multiplex requires CUDA')
    return Sam31(checkpoint, torch)


def embedding_image(rgb, mask=None, side=224):
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        rows = slice(ys.min(), ys.max()+1)
        cols = slice(xs.min(), xs.max()+1)
        rgb = np.where(mask[rows, cols][..., None], rgb[rows, cols], 127).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.thumbnail((side, side), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (side, side), (127, 127, 127))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return canvas


class DinoV3:
    def __init__(self, directory, device, torch):
        from transformers import AutoImageProcessor, AutoModel

        self.torch, self.device = torch, device
        path = Path(directory).resolve()
        files = ('config.json', 'preprocessor_config.json', 'model.safetensors')
        if not all((path/name).is_file() for name in files):
            raise ValueError('Use the downloaded local DINOv3 ViT-S/16 safetensors snapshot')
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True,
                                                             trust_remote_code=False, use_fast=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True).to(device).eval()
        if self.model.config.model_type != 'dinov3_vit' or self.model.config.patch_size != 16 or self.model.config.hidden_size != 384:
            raise ValueError('Expected a DINOv3 ViT-S/16 encoder compatible with 384-dimensional features')
        self.model.requires_grad_(False)
        self.metadata = dict(method='dinov3_vits16', transformers_version=version('transformers'),
                             pooling='CLS, L2 normalization', input_size=224,
                             files_sha256={name: sha256_file(path/name) for name in files})

    def __call__(self, image):
        crop = {'do_center_crop': False} if getattr(self.processor, 'do_center_crop', False) else {}
        inputs = self.processor(images=image, do_resize=False, return_tensors='pt', **crop)
        with self.torch.inference_mode():
            output = self.model(**inputs.to(self.device)).pooler_output
            vector = self.torch.nn.functional.normalize(output.float(), dim=-1)[0].cpu().numpy()
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError('Invalid DINOv3 embedding')
        return vector.tolist()
