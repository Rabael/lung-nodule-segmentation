"""Data loading, splitting and transforms for MSD Task06 Lung."""
import json
from pathlib import Path

import numpy as np
from monai import transforms as T


def list_cases(data_dir):
    """Return a sorted list of {"image": ..., "label": ...} dicts."""
    data_dir = Path(data_dir)
    images = sorted(
        p for p in (data_dir / "imagesTr").glob("*.nii.gz")
        if not p.name.startswith(".")
    )
    cases = []
    for img in images:
        lab = data_dir / "labelsTr" / img.name
        if not lab.exists():
            raise FileNotFoundError(f"No mask for {img.name}")
        cases.append({"image": str(img), "label": str(lab)})
    return cases


def make_splits(cases, val_frac=0.15, test_frac=0.15, seed=42):
    """Random train/val/test split at the scan level, reproducible via seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases))
    n_test = int(round(len(cases) * test_frac))
    n_val = int(round(len(cases) * val_frac))
    test = [cases[i] for i in order[:n_test]]
    val = [cases[i] for i in order[n_test:n_test + n_val]]
    train = [cases[i] for i in order[n_test + n_val:]]
    return {"train": train, "val": val, "test": test}


def save_splits(splits, path):
    """Store only file names, so the split works on any machine."""
    out = {k: [Path(c["image"]).name for c in v] for k, v in splits.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))


def load_splits(path, data_dir):
    data_dir = Path(data_dir)
    names = json.loads(Path(path).read_text())
    return {
        k: [
            {
                "image": str(data_dir / "imagesTr" / n),
                "label": str(data_dir / "labelsTr" / n),
            }
            for n in v
        ]
        for k, v in names.items()
    }


def get_transforms(train, patch_size=(96, 96, 96), pixdim=(1.5, 1.5, 2.0),
                   num_samples=2):
    """MONAI pipeline. Values are provisional until the exploration is done."""
    keys = ["image", "label"]
    steps = [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
        T.ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
    if train:
        steps += [
            T.RandCropByPosNegLabeld(
                keys=keys, label_key="label", spatial_size=patch_size,
                pos=1, neg=1, num_samples=num_samples,
                image_key="image", image_threshold=0,
            ),
            T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    steps.append(T.EnsureTyped(keys=keys))
    return T.Compose(steps)