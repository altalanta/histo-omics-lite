from __future__ import annotations

import torch

from histo_omics_lite.models.multimodal import MultimodalClipModule
from histo_omics_lite.models.simclr import SimCLRModule


def test_simclr_forward_backward() -> None:
    model = SimCLRModule(projection_dim=32, projection_hidden_dim=64)
    batch = {
        "view1": torch.randn(4, 3, 224, 224),
        "view2": torch.randn(4, 3, 224, 224),
    }
    loss = model.training_step(batch, 0)
    loss.backward()
    assert loss.item() > 0


def test_multimodal_clip_training_step() -> None:
    module = MultimodalClipModule(omics_dim=16, embed_dim=32, image_hidden_dim=64, omics_hidden_dim=64)
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "omics": torch.randn(4, 16),
    }
    loss = module.training_step(batch, 0)
    loss.backward()
    assert loss.item() > 0
