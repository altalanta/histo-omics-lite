from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from histo_omics_lite.data.loader import HistoOmicsDataset
from histo_omics_lite.data.synthetic import make_tiny
from histo_omics_lite.training.train import (
    DataConfig,
    HistoOmicsModule,
    OptimizerConfig,
    TrainConfig,
    TrainerConfig,
    run_training,
)
from histo_omics_lite.utils.determinism import hash_embeddings, set_determinism


def _compute_digest(checkpoint_path: Path, dataset_dir: Path) -> str:
    set_determinism(11)
    dataset = HistoOmicsDataset(
        image_root=dataset_dir / "tiles",
        omics_csv=dataset_dir / "omics.csv",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = HistoOmicsModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    image_embeddings = []
    with torch.no_grad():
        for batch in loader:
            img_emb, _ = model.encode_batch({"image": batch["image"], "omics": batch["omics"]})
            image_embeddings.append(img_emb)

    embedding_tensor = torch.cat(image_embeddings)[:10]
    return hash_embeddings(embedding_tensor)


def _make_config(dataset_dir: Path, output_dir: Path) -> TrainConfig:
    return TrainConfig(
        seed=11,
        temperature=0.07,
        omics_input_dim=50,
        output_dir=str(output_dir),
        data=DataConfig(
            root=str(dataset_dir),
            batch_size=4,
            num_workers=0,
            val_fraction=0.25,
            use_webdataset=False,
        ),
        optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0),
        trainer=TrainerConfig(max_epochs=1, limit_train_batches=10, limit_val_batches=2),
    )


def test_training_hash_is_stable(tmp_path: Path) -> None:
    """Test that training produces deterministic results with fixed seeds."""
    dataset_dir = tmp_path / "synthetic"
    make_tiny(dataset_dir)

    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    ckpt1 = run_training(_make_config(dataset_dir, run1_dir))
    ckpt2 = run_training(_make_config(dataset_dir, run2_dir))

    digest1 = _compute_digest(Path(ckpt1), dataset_dir)
    digest2 = _compute_digest(Path(ckpt2), dataset_dir)

    # Both runs should produce identical hashes due to fixed seeds
    assert digest1 == digest2

    # The hash should match the expected value for these specific parameters
    expected_hash = "32d359d832015c54b690a0f87047c7d4a8e511e95afa7f777c35e7faca58bf32"
    assert digest1 == expected_hash
