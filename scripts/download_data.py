"""Download MSD Task06_Lung (training images and labels) into data/raw/."""
import sys
import tarfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "msd-for-monai-eu"
REGION = "eu-west-2"
PREFIX = "Task06_Lung"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
WANTED = ("imagesTr/", "labelsTr/", "dataset.json")


def list_keys(s3, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download(s3, key, size, dest):
    if dest.exists() and dest.stat().st_size == size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dest) + ".part")
    s3.download_file(BUCKET, key, str(part))
    part.rename(dest)
    return True


def is_hidden(key):
    return Path(key).name.startswith(".")


def main():
    s3 = boto3.client(
        "s3", region_name=REGION, config=Config(signature_version=UNSIGNED)
    )
    keys = list(list_keys(s3, PREFIX))

    if not keys:
        print(f"Nothing found under prefix {PREFIX}. Top level of the bucket:")
        resp = s3.list_objects_v2(Bucket=BUCKET, Delimiter="/")
        for p in resp.get("CommonPrefixes", []):
            print("  ", p["Prefix"])
        for o in resp.get("Contents", [])[:20]:
            print("  ", o["Key"])
        sys.exit(1)

    # Case 1: uncompressed layout (Task06_Lung/imagesTr/...)
    files = [
        (k, s) for k, s in keys
        if any(w in k for w in WANTED) and not is_hidden(k)
        and not k.endswith((".tar", ".tar.gz"))
    ]
    if files:
        print(f"Found {len(files)} files to download")
        for i, (key, size) in enumerate(files, 1):
            new = download(s3, key, size, DEST / key)
            print(f"[{i}/{len(files)}] {key} {'downloaded' if new else 'already there'}")
    else:
        # Case 2: a tar or tar.gz archive
        archives = [(k, s) for k, s in keys if k.endswith((".tar", ".tar.gz"))]
        if not archives:
            print("Found keys but no usable layout:")
            for k, _ in keys[:20]:
                print("  ", k)
            sys.exit(1)
        key, size = archives[0]
        archive = DEST / key
        print(f"Downloading archive {key} ({size / 1e9:.1f} GB)")
        download(s3, key, size, archive)
        with tarfile.open(archive, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if any(w in m.name for w in WANTED) and not is_hidden(m.name)
            ]
            print(f"Extracting {len(members)} files")
            tar.extractall(DEST, members=members, filter="data")

    for sub in ("imagesTr", "labelsTr"):
        n = len(list((DEST / PREFIX / sub).glob("*.nii.gz")))
        print(f"{sub}: {n} files")


if __name__ == "__main__":
    main()