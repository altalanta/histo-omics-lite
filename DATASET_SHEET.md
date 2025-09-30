# Dataset Sheet – histo-omics-lite synthetic set

## Summary
- **Dataset name:** histo-omics-lite synthetic tiles.
- **Responsible party:** Altalanta engineering (for questions: engineering@altalanta.ai).
- **Data modality:** 64 RGB image tiles paired with 50-dimensional omics vectors and class labels.
- **Creation process:** Fully synthetic, generated locally via `histo_omics_lite.data.synthetic.make_tiny`.

## Motivation
- Provide a deterministic, license-clean dataset for testing multimodal workflows.
- Enable fast CI checks without external downloads or privacy risk.

## Composition
- 4 balanced pseudo-classes (16 tiles each) with deterministic naming (`tile_000` … `tile_063`).
- Omics CSV includes `tile_id`, `label`, and 50 gene columns (`gene_000` … `gene_049`).
- Pixel patterns encode simple sinusoidal variations plus diagonal markers for debugging.

## Collection process
- Images are procedurally generated with seeded NumPy noise and sinusoidal bands.
- Omics vectors derive from seeded Gaussian draws conditioned on the class label.
- All artefacts are written under the requested output directory (e.g. `data/synthetic`).

## Uses
- CI smoke runs (`make smoke`) and pytest fixtures for the histo-omics-lite project.
- Demonstrations of deterministic data generation for ML pipelines.

## Limitations
- No biological realism; values are illustrative only.
- Small scale (64 samples) – not representative of production workloads.
- Not intended for model validation beyond ensuring code paths execute end-to-end.

## Distribution
- Generated locally; no redistribution required.
- Licensed under the same project license (Apache-2.0).

## Maintenance
- Regenerate as needed; rerunning `make data` overwrites the `data/synthetic` directory deterministically.
