"""Inference CLI for embedding new tiles and omics vectors."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from histo_omics_lite.data.transforms import get_eval_transform
from histo_omics_lite.data.utils import seed_worker
from histo_omics_lite.models.multimodal import MultimodalClipModule

app = typer.Typer(help="Embed histology tiles and transcriptomic features.")


class TileDataset(Dataset):
    def __init__(self, files: List[Path], image_size: int = 224) -> None:
        self.files = files
        self.transform = get_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return {"image": tensor, "sample_id": path.stem}


def _load_omics_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        sep = "," if path.suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported omics format: {path}")


@app.command()
def embed(
    checkpoint: Path = typer.Option(..., help="Path to a trained CLIP checkpoint"),
    tiles_dir: Path = typer.Option(..., help="Directory containing tile images"),
    omics_table: Path = typer.Option(..., help="Parquet/CSV with sample_id + omics features"),
    output_dir: Path = typer.Option(Path("outputs/inference"), help="Directory to store embeddings"),
    batch_size: int = typer.Option(16, help="Inference batch size"),
    device: str = typer.Option("cpu", help="Torch device"),
    image_size: int = typer.Option(224, help="Resize for tiles"),
) -> None:
    """Embed tiles and omics vectors, writing Parquet + NPZ outputs."""
    files = sorted([p for p in tiles_dir.glob("*.png")])
    if not files:
        raise typer.BadParameter(f"No PNG files found in {tiles_dir}")

    dataset = TileDataset(files, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    model = MultimodalClipModule.load_from_checkpoint(str(checkpoint))
    model.eval()
    model = model.to(torch.device(device))

    omics_df = _load_omics_table(omics_table)
    if "sample_id" not in omics_df.columns:
        raise typer.BadParameter("omics table must include a 'sample_id' column")
    omics_df = omics_df.set_index("sample_id")

    image_embeddings: List[torch.Tensor] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["sample_id"]
            embeddings = model.forward_image(images)
            image_embeddings.append(embeddings.cpu())
            sample_ids.extend(ids)

    image_matrix = torch.cat(image_embeddings, dim=0).numpy()
    embedding_map = {sid: emb for sid, emb in zip(sample_ids, image_matrix)}
    keep_ids = [sid for sid in sample_ids if sid in omics_df.index]
    if not keep_ids:
        raise typer.BadParameter("No overlapping sample_ids between tiles and omics table")

    image_matrix = np.stack([embedding_map[sid] for sid in keep_ids])
    omics_matrix = omics_df.loc[keep_ids].to_numpy(dtype=np.float32)
    with torch.no_grad():
        omics_embeddings = (
            model.forward_omics(torch.from_numpy(omics_matrix).to(device)).cpu().numpy()
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_df = pd.DataFrame(image_matrix, columns=[f"dim_{i}" for i in range(image_matrix.shape[1])])
    image_df.insert(0, "sample_id", keep_ids)
    omics_df_out = pd.DataFrame(omics_embeddings, columns=[f"dim_{i}" for i in range(omics_embeddings.shape[1])])
    omics_df_out.insert(0, "sample_id", keep_ids)

    image_df.to_parquet(output_dir / "image_embeddings.parquet", index=False)
    omics_df_out.to_parquet(output_dir / "omics_embeddings.parquet", index=False)

    np.savez(output_dir / "embeddings.npz", sample_id=keep_ids, image=image_matrix, omics=omics_embeddings)
    typer.echo(f"Wrote embeddings for {len(keep_ids)} samples to {output_dir}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
