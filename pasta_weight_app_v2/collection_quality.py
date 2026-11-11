"""Original-image diagnostics, never weight predictions or training approvals."""
from io import BytesIO
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pasta_weight_app_v2.experiments.marker_reference import analyze_marker_reference
from pasta_weight_app_v2.experiments.stability_pipeline import process, configurations


def inspect_photo(data, *, reference=False, printed=False, coplanar=False, inherited_flags=()):
    # Validate originals before decoding, including callers outside the UI.
    if not isinstance(data, bytes) or not data or len(data) > 20 * 1024 * 1024:
        raise ValueError('20MB以内のJPEG・PNG・WebP画像を選んでください。')
    try:
        with Image.open(BytesIO(data)) as im:
            if (im.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(im, 'n_frames', 1) != 1
                    or min(im.size) < 16 or im.width * im.height > 40_000_000):
                raise ValueError('静止画の16px以上・4,000万画素以内の画像を選んでください。')
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('写真を読み込めません。別の画像を選んでください。') from exc
    features, scene, mask=process(rgb, next(c for c in configurations() if c.name=='B_normalized5'))
    gray=cv2.cvtColor(scene,cv2.COLOR_RGB2GRAY)
    flags=set(inherited_flags) | set(features['qc_flags'])
    if gray.mean()<35:flags.add('dark_image')
    if cv2.Laplacian(gray,cv2.CV_32F).var()<10:flags.add('blur_review')
    meta, _, overlay=analyze_marker_reference(rgb,print_scale_verified=printed,coplanar_confirmed=coplanar)
    if reference and not meta['usable_for_metric_scale']:flags.add('reference_unavailable')
    severe=flags & {'foreground_touches_frame','dark_image','reference_unavailable','insufficient_foreground'}
    return dict(status='撮り直し推奨' if severe else '確認が必要', flags=sorted(flags),
                whole_pasta_verified=False, training_approved=False,
                reference=meta, blur_variance=float(cv2.Laplacian(gray,cv2.CV_32F).var())), overlay
