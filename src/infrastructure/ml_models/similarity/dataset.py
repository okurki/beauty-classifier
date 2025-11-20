# src/ml_models/similarity/dataset.py
import os
import json
import hashlib
from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Корень датасета: внутри — папки по именам знаменитостей
CELEB_DATA_PATH = "datasets/open_famous_people_faces"
CLASSES_FILE = os.path.join(CELEB_DATA_PATH, "classes.json")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_classes(root: str) -> List[str]:
    """Список классов = имена подпапок (отсортированный, стабильный)."""
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    return classes


def _list_images(root: str, cls: str) -> List[str]:
    """Вернёт список относительных путей 'класс/файл' (отсортированный)."""
    folder = os.path.join(root, cls)
    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(os.path.join(cls, name))
    files.sort()
    return files


def _hash_split(relpath: str, seed: int) -> float:
    """
    Детерминированная псевдослучайная величина в [0,1) по относительному пути и seed.
    Позволяет делать стабильный сплит без хранения манифестов.
    """
    h = hashlib.md5((relpath + str(seed)).encode("utf-8")).hexdigest()
    # возьмём первые 8 hex-символов как uint32
    val = int(h[:8], 16)
    return val / 0xFFFFFFFF


class CelebrityFolderDataset(Dataset):
    """
    Датасет по папочной структуре:
      datasets/open_famous_people_faces/
        ├─ aaron_taylor_johnson/
        │    ├─ face_detected_01ae6051.jpg
        │    └─ ...
        ├─ tom_hanks/
        └─ ...

    split: 'train' или 'val'
    Сплит делается детерминированно (per-file) по хэшу пути и seed, баланс по классам сохраняется
    автоматически, т.к. решение принимается для каждого файла внутри класса.
    """

    def __init__(
        self,
        root: str = CELEB_DATA_PATH,
        split: str = "train",
        transform=None,
        val_ratio: float = 0.1,
        seed: int = 42,
        label2id: Optional[Dict[str, int]] = None,
        classes_file: Optional[str] = None,
    ):
        assert split in {"train", "val"}
        assert 0.0 < val_ratio < 1.0

        self.root = root
        self.split = split
        self.transform = transform
        self.val_ratio = val_ratio
        self.seed = seed

        # Маппинг классов
        self._classes_file = classes_file or CLASSES_FILE
        if label2id is not None:
            self.label2id = label2id
        else:
            if os.path.exists(self._classes_file):
                with open(self._classes_file, "r", encoding="utf-8") as f:
                    id2label = json.load(f)
                # в файле ключи — строки id
                self.label2id = {v: int(k) for k, v in id2label.items()}
            else:
                cls_names = _iter_classes(self.root)
                self.label2id = {name: i for i, name in enumerate(cls_names)}
                id2label = {str(i): name for name, i in self.label2id.items()}
                with open(self._classes_file, "w", encoding="utf-8") as f:
                    json.dump(id2label, f, ensure_ascii=False, indent=2)

        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        # Индекс изображений для нужного split
        self.samples: List[Tuple[str, int]] = []
        for lbl_name in self.label2id.keys():
            files = _list_images(self.root, lbl_name)
            for relpath in files:
                r = _hash_split(relpath, seed=self.seed)
                is_val = r < self.val_ratio
                if (self.split == "val" and is_val) or (
                    self.split == "train" and not is_val
                ):
                    self.samples.append((relpath, self.label2id[lbl_name]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for split='{self.split}'. "
                f"Check '{self.root}' structure and extensions: {sorted(IMG_EXTS)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    def __getitem__(self, idx: int):
        relpath, label = self.samples[idx]
        img_path = os.path.join(self.root, relpath)

        with Image.open(img_path) as image:
            if self.transform is not None:
                image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor


def get_data_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Возвращает (train_loader, val_loader) из папочной структуры.
    label2id фиксируем от train, чтобы индексы совпадали.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Train сначала — формируем label2id и сохраняем classes.json
    train_ds = CelebrityFolderDataset(
        root=CELEB_DATA_PATH,
        split="train",
        transform=train_transform,
        val_ratio=val_ratio,
        seed=seed,
        classes_file=CLASSES_FILE,
    )

    # Val — используем тот же маппинг индексов
    val_ds = CelebrityFolderDataset(
        root=CELEB_DATA_PATH,
        split="val",
        transform=val_transform,
        val_ratio=val_ratio,
        seed=seed,
        label2id=train_ds.label2id,
        classes_file=CLASSES_FILE,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
