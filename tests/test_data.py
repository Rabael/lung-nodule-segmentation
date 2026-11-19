import nibabel as nib
import numpy as np

from lungseg.data import get_transforms, list_cases, make_splits


def _write_case(folder, name, shape=(48, 48, 32)):
    rng = np.random.default_rng(0)
    ct = rng.normal(-500, 200, shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:28, 20:28, 12:20] = 1
    (folder / "imagesTr").mkdir(parents=True, exist_ok=True)
    (folder / "labelsTr").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(ct, np.eye(4)), folder / "imagesTr" / name)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), folder / "labelsTr" / name)


def test_list_cases(tmp_path):
    for i in range(3):
        _write_case(tmp_path, f"lung_{i:03d}.nii.gz")
    cases = list_cases(tmp_path)
    assert len(cases) == 3
    assert cases[0]["image"].endswith("lung_000.nii.gz")


def test_make_splits():
    cases = [{"image": f"i{i}", "label": f"l{i}"} for i in range(20)]
    s = make_splits(cases, val_frac=0.15, test_frac=0.15, seed=1)
    names = [c["image"] for part in s.values() for c in part]
    assert len(names) == 20 and len(set(names)) == 20
    assert s == make_splits(cases, val_frac=0.15, test_frac=0.15, seed=1)


def test_val_transforms(tmp_path):
    _write_case(tmp_path, "lung_000.nii.gz")
    case = list_cases(tmp_path)[0]
    out = get_transforms(train=False, pixdim=(1, 1, 1))(case)
    img, lab = np.asarray(out["image"]), np.asarray(out["label"])
    assert img.shape == (1, 48, 48, 32)
    assert img.min() >= 0 and img.max() <= 1
    assert set(np.unique(lab)) <= {0, 1}


def test_train_transforms(tmp_path):
    _write_case(tmp_path, "lung_000.nii.gz")
    case = list_cases(tmp_path)[0]
    tf = get_transforms(
        train=True, patch_size=(16, 16, 16), pixdim=(1, 1, 1), num_samples=2
    )
    patches = tf(case)
    assert len(patches) == 2
    assert np.asarray(patches[0]["image"]).shape == (1, 16, 16, 16)