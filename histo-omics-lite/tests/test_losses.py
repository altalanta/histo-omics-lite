from __future__ import annotations

import torch

from histo_omics_lite.models.losses import clip_contrastive_loss, nt_xent_loss


def test_nt_xent_loss_symmetry() -> None:
    torch.manual_seed(0)
    z_i = torch.randn(8, 128)
    z_j = torch.randn(8, 128)
    loss = nt_xent_loss(z_i, z_j, temperature=0.1)
    assert loss.item() > 0


def test_clip_contrastive_loss_basic() -> None:
    torch.manual_seed(0)
    img = torch.randn(4, 256)
    omics = torch.randn(4, 256)
    loss = clip_contrastive_loss(img, omics, temperature=0.2)
    assert loss.item() > 0
