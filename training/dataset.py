from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


FRUIT_ORDER = ["Apple", "Banana", "Grape", "Mango", "Orange"]
CONDITION_ORDER = ["Fresh", "Rotten", "Formalin-mixed"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    split_source: str
    fruit: str
    condition: str
    label_name: str
    source_type: str
    path: str
    archive_path: str | None = None
    inner_path: str | None = None


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _iter_local_images(folder: Path) -> list[Path]:
    return sorted([path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])


def _collect_original_records(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for fruit in FRUIT_ORDER:
        fruit_dir = dataset_root / fruit
        if not fruit_dir.exists():
            continue
        for condition in CONDITION_ORDER:
            condition_dir = fruit_dir / condition
            if not condition_dir.exists():
                continue
            for image_path in _iter_local_images(condition_dir):
                label_name = f"{fruit}__{condition}"
                records.append(
                    SampleRecord(
                        split_source="original",
                        fruit=fruit,
                        condition=condition,
                        label_name=label_name,
                        source_type="directory",
                        path=str(image_path),
                    )
                )
    return records


def _collect_augmentation_records(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for fruit in FRUIT_ORDER:
        fruit_dir = dataset_root / fruit
        if not fruit_dir.exists():
            continue
        zip_by_condition: dict[str, Path] = {}
        dir_by_condition: dict[str, Path] = {}
        for item in fruit_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".zip":
                zip_by_condition[item.stem] = item
            elif item.is_dir():
                dir_by_condition[item.name] = item

        for condition in CONDITION_ORDER:
            label_name = f"{fruit}__{condition}"
            if condition in zip_by_condition:
                with zipfile.ZipFile(zip_by_condition[condition]) as archive:
                    names = sorted(
                        [
                            info.filename
                            for info in archive.infolist()
                            if not info.is_dir() and _is_image_name(info.filename)
                        ]
                    )
                for inner_path in names:
                    records.append(
                        SampleRecord(
                            split_source="augmentation",
                            fruit=fruit,
                            condition=condition,
                            label_name=label_name,
                            source_type="zip",
                            path=f"{zip_by_condition[condition]}::{inner_path}",
                            archive_path=str(zip_by_condition[condition]),
                            inner_path=inner_path,
                        )
                    )
            elif condition in dir_by_condition:
                for image_path in _iter_local_images(dir_by_condition[condition]):
                    records.append(
                        SampleRecord(
                            split_source="augmentation",
                            fruit=fruit,
                            condition=condition,
                            label_name=label_name,
                            source_type="directory",
                            path=str(image_path),
                        )
                    )
    return records


def collect_records(data_root: str | Path, variant: str) -> list[SampleRecord]:
    data_root = Path(data_root)
    original_root = data_root / "Fruits Original"
    augmentation_root = data_root / "Fruits Augmentation"

    original_records = _collect_original_records(original_root)
    augmentation_records = _collect_augmentation_records(augmentation_root)

    if variant == "original":
        return original_records
    if variant == "augmentation":
        return augmentation_records
    if variant == "combined":
        return original_records + augmentation_records
    raise ValueError("variant must be one of: original, augmentation, combined")


def limit_records_per_class(records: list[SampleRecord], limit_per_class: int | None, seed: int) -> list[SampleRecord]:
    if not limit_per_class or limit_per_class <= 0:
        return records

    rng = random.Random(seed)
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.label_name, []).append(record)

    limited: list[SampleRecord] = []
    for label_name, items in grouped.items():
        if len(items) <= limit_per_class:
            limited.extend(items)
        else:
            limited.extend(rng.sample(items, limit_per_class))
    return limited


def build_class_names(records: list[SampleRecord]) -> list[str]:
    seen = {record.label_name for record in records}
    return [
        f"{fruit}__{condition}"
        for fruit in FRUIT_ORDER
        for condition in CONDITION_ORDER
        if f"{fruit}__{condition}" in seen
    ]


def split_records(
    records: list[SampleRecord],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[SampleRecord]]:
    labels = [record.label_name for record in records]
    train_records, temp_records = train_test_split(
        records,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [record.label_name for record in temp_records]
    relative_test_size = test_size / (val_size + test_size)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels,
    )
    return {"train": train_records, "val": val_records, "test": test_records}


class FruitClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        class_to_idx: dict[str, int],
        transform: Any | None = None,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self._zip_handles: dict[str, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _open_image(self, record: SampleRecord) -> Image.Image:
        if record.source_type == "zip":
            assert record.archive_path and record.inner_path
            if record.archive_path not in self._zip_handles:
                self._zip_handles[record.archive_path] = zipfile.ZipFile(record.archive_path)
            image_bytes = self._zip_handles[record.archive_path].read(record.inner_path)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.load()
            return image
        image = Image.open(record.path).convert("RGB")
        image.load()
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._open_image(record)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": self.class_to_idx[record.label_name],
            "label_name": record.label_name,
            "fruit": record.fruit,
            "condition": record.condition,
            "source": record.split_source,
            "path": record.path,
        }
