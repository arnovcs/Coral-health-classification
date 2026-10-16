# Data

## `coralscapes/` (not committed)

Local cache of the [Coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) dataset, written by `notebooks/01_download_data.ipynb` with `save_to_disk` and read by the other notebooks with `load_from_disk`. It is about 2,000 images at 2048 × 1024, so it is git-ignored.

> The notebooks currently use hardcoded absolute paths. Point them at `data/coralscapes/` (for example `load_from_disk("../data/coralscapes")` when running from `notebooks/`).

## `stats/` (committed)

Summaries produced by `notebooks/02_explore_data.ipynb`:

| File | Contents |
|---|---|
| `coralscapes_id_counts.csv` | Pixel count per class ID (rows) for each split (columns) |
| `coralscapes_image_sizes.csv` | Width, height and area of every image |
| `coralscapes_coocc_train.csv` | Image-level class co-occurrence matrix for the train split |
