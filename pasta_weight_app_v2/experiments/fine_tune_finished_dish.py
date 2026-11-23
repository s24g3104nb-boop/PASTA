"""Research-only fine-tuning of the final visual block of the pinned OpenCLIP model."""
from collections import Counter

from PIL import Image
import numpy as np

from ..dish_classifier import LinearDishHead, STANDARD_DISH_NAMES
from ..pasta_vision import image_to_rgb

EPOCHS = 5
BATCH_SIZE = 8
BLOCK_LR = 1e-5
HEAD_LR = .003


def _weights(examples):
    photos_by_group = Counter((item.session_id, item.label) for item in examples)
    groups_by_class = Counter(label for _, label in photos_by_group)
    return [len(examples) / (len(groups_by_class) * groups_by_class[item.label]
                             * photos_by_group[item.session_id, item.label]) for item in examples]


def adapt_visual_block(classifier, seed_head, examples):
    """Fit on the supplied training split only and return a local research artifact."""
    torch = classifier.torch
    model = classifier.model
    block = model.visual.transformer.resblocks[-1]
    block.requires_grad_(True)
    head = torch.nn.Linear(seed_head.weights.shape[1], len(STANDARD_DISH_NAMES),
                           device=classifier.device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(seed_head.weights))
        head.bias.copy_(torch.as_tensor(seed_head.bias))
    opt = torch.optim.AdamW([dict(params=block.parameters(), lr=BLOCK_LR),
                             dict(params=head.parameters(), lr=HEAD_LR)], weight_decay=.01)
    pixels = torch.stack([classifier.preprocess(Image.fromarray(image_to_rgb(item.path)))
                          for item in examples])
    targets = torch.tensor([STANDARD_DISH_NAMES.index(item.label) for item in examples])
    weights = torch.tensor(_weights(examples), dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    for _ in range(EPOCHS):
        for ids in torch.randperm(len(examples), generator=generator).split(BATCH_SIZE):
            opt.zero_grad(set_to_none=True)
            embedding = model.encode_image(pixels[ids].to(classifier.device), normalize=True)
            losses = torch.nn.functional.cross_entropy(
                head(embedding), targets[ids].to(classifier.device), reduction='none')
            loss = (losses * weights[ids].to(classifier.device)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite OpenCLIP fine-tuning loss')
            loss.backward()
            opt.step()
    block.requires_grad_(False)
    head = LinearDishHead(head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy())
    block_state = {key: tensor.detach().cpu() for key, tensor in block.state_dict().items()}
    return head, block_state


def adapted_predictions(classifier, head, examples, scope_labels):
    predictions = []
    for example, scope in zip(examples, scope_labels, strict=True):
        if scope is None:
            predictions.append(None)
            continue
        image = Image.fromarray(image_to_rgb(example.path))
        features = classifier.encode(image)[0].float().cpu().numpy()
        predictions.append(head.select(np.asarray(features), scope))
    return predictions
