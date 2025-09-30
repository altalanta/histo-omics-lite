from __future__ import annotations

import torch

from histo_omics_lite.models import ContrastiveHead, OmicsEncoder, VisionEncoder


def test_contrastive_shapes_and_values() -> None:
    torch.manual_seed(0)
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64)
    omics = torch.randn(batch_size, 50)

    vision = VisionEncoder(output_dim=128)
    omics_encoder = OmicsEncoder(input_dim=50, hidden_dim=128, output_dim=128)
    head = ContrastiveHead(temperature=0.07)

    image_embeddings = vision(images)
    omics_embeddings = omics_encoder(omics)

    loss, logits, norm_img, norm_omics = head(image_embeddings, omics_embeddings)

    assert image_embeddings.shape == (batch_size, 128)
    assert omics_embeddings.shape == (batch_size, 128)
    assert logits.shape == (batch_size, batch_size)
    assert torch.isfinite(loss), "Loss contains NaNs or infs"
    assert torch.isclose(norm_img.norm(dim=1), torch.ones(batch_size), atol=1e-5).all()
    assert torch.isclose(norm_omics.norm(dim=1), torch.ones(batch_size), atol=1e-5).all()
