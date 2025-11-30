import io
import os
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from ..base_model import ModelBase
from src.interfaces.api.v1.schemas.similarity import SimilarityPrediction
from src.config import config

try:
    import mlflow
    import torchmetrics
    from tqdm import tqdm

    from .dataset import get_data_loaders
except ImportError:
    pass

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)
CLASSES_FILE = "/classes.json"


def _load_id2label(path: str) -> Optional[Dict[int, str]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


class CelebrityMatcherModel(ModelBase):
    _name: str = "celebrity_matcher"

    def __init__(
        self,
        out_features: int = 512,
        top_k: int = 5,
        classes_file: Optional[str] = None,
    ):
        super().__init__()
        self.top_k = top_k

        self._transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.CenterCrop((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._classes_file = (
            classes_file or config.ml.celebrities_dataset_dir + CLASSES_FILE
        )  # datasets/open_famous_people_faces/classes.json
        self.id2label = _load_id2label(self._classes_file)
        self.num_classes = len(self.id2label) if self.id2label else None

        backbone = self._load_vggface2_backbone()

        for p in backbone.parameters():
            p.requires_grad = False

        in_features = 1792
        classifier_out = self.num_classes if self.num_classes else 2
        backbone.classify = True

        backbone.last_linear = nn.Linear(in_features, out_features, bias=False)
        backbone.last_bn = nn.BatchNorm1d(
            out_features, eps=0.001, momentum=0.1, affine=True
        )

        # the final classification layer
        backbone.logits = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, classifier_out),
        )

        self._model = backbone
        self._model.to(self._device)
        self.loaded = False

    def _load_vggface2_backbone(self):
        try:
            logger.info(
                "Loading VGGFace2 pretrained InceptionResnetV1 from facenet-pytorch"
            )
            return InceptionResnetV1(pretrained="vggface2", classify=False)
        except Exception as e:
            logger.warning(
                f"Could not load VGGFace2: {e}. Falling back to CASIA-WebFace"
            )
            try:
                return InceptionResnetV1(pretrained="casia-webface", classify=False)
            except Exception as e2:
                logger.error(f"Could not load any face recognition model: {e2}")
                raise

    def _setup_metrics(self, device, num_classes: int):
        acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )
        acc5 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        ).to(device)
        f1m = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        return nn.ModuleDict({"acc1": acc1, "acc5": acc5, "f1_macro": f1m})

    def train(
        self,
        epochs: int = 10,
        lr: float = 1e-3,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        if not self.loaded:
            try:
                self.load()
            except Exception:
                logger.warning("No existing checkpoint. Training from scratch.")

        train_loader, val_loader = get_data_loaders()
        num_classes = train_loader.dataset.num_classes

        if hasattr(train_loader.dataset, "id2label"):
            self.id2label = train_loader.dataset.id2label

        if optimizer is None:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=5
            )

        metrics = self._setup_metrics(self._device, num_classes)

        best_acc = 0.0

        mlflow.set_experiment("Training CelebrityMatcher model")

        mlflow.start_run(
            run_name=f"CelebrityMatcher {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": scheduler.__class__.__name__ if scheduler else "None",
                "model_name": "vggface2 inceptionresnetv1",
                "out_features": 512,
                "fine_tuning": "feature_extraction",
                "num_classes": num_classes,
                "top_k": self.top_k,
            }
        )
        mlflow.set_tag(
            "model_architecture", "vggface2_inceptionresnetv1_frozen_fc_custom"
        )
        mlflow.log_artifact(__file__, artifact_path="code")

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for phase, loader in [("train", train_loader), ("val", val_loader)]:
                self._model.train(mode=(phase == "train"))

                # reset metrics
                for m in metrics.values():
                    m.reset()

                # reset metrics
                for m in metrics.values():
                    m.reset()

                running_loss = 0.0
                total = 0

                pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch + 1}")
                logger.debug(f"Phase {phase.capitalize()}")
                for images, labels in pbar:
                    images = images.to(self._device)
                    labels = labels.to(self._device)
                    bs = images.size(0)
                    total += bs

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        logits = self._model(images)
                        loss = criterion(logits, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * bs

                    preds = torch.argmax(logits, dim=1)
                    metrics["acc1"].update(preds, labels)  # top-1 по предсказаниям
                    metrics["acc5"].update(logits, labels)  # top-5 по логитам
                    metrics["f1_macro"].update(preds, labels)

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc1": f"{metrics['acc1'].compute().item():.4f}",
                        }
                    )

                epoch_loss = running_loss / max(1, total)
                computed = {name: m.compute().item() for name, m in metrics.items()}

                mlflow.log_metrics(
                    {
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_acc1": computed["acc1"],
                        f"{phase}_acc5": computed["acc5"],
                        f"{phase}_f1_macro": computed["f1_macro"],
                    },
                    step=epoch,
                )

                logger.info(
                    f"{phase}: loss={epoch_loss:.4f} acc1={computed['acc1']:.4f} "
                    f"acc5={computed['acc5']:.4f} f1_macro={computed['f1_macro']:.4f}"
                )

                if phase == "val":
                    if computed["acc1"] > best_acc:
                        best_acc = computed["acc1"]
                        self.save()
                        try:
                            mlflow.pytorch.log_model(
                                self._model,
                                "best_model",
                                registered_model_name=self._name,
                            )
                        except Exception:
                            pass
                        mlflow.log_metric("best_val_acc1", best_acc)
                        logger.info(
                            f"New best model saved with val acc1: {best_acc:.4f}"
                        )

                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(computed["acc1"])
                        mlflow.log_metric(
                            "learning_rate", optimizer.param_groups[0]["lr"], step=epoch
                        )

        mlflow.log_metric("final_best_val_acc1", best_acc)
        mlflow.set_tag("best_model", "True")
        mlflow.end_run()
        logger.info(f"Training complete. Best val acc@1: {best_acc:.4f}")

    def evaluate(self) -> dict:
        if not self.loaded:
            self.load()

        self._model.eval()
        # используем валид. лоадер как тестовый, т.к. сплит двухчастный
        _, test_loader = get_data_loaders()
        num_classes = test_loader.dataset.num_classes
        metrics = self._setup_metrics(self._device, num_classes)

        running_loss = 0.0
        total = 0
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(test_loader, desc="Evaluating", leave=True)
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self._device)
                labels = labels.to(self._device)
                bs = images.size(0)
                total += bs

                logits = self._model(images)
                loss = criterion(logits, labels)
                running_loss += loss.item() * bs

                preds = torch.argmax(logits, dim=1)
                metrics["acc1"].update(preds, labels)
                metrics["acc5"].update(logits, labels)
                metrics["f1_macro"].update(preds, labels)

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc1": f"{metrics['acc1'].compute().item():.4f}",
                    }
                )

        final = {name: m.compute().item() for name, m in metrics.items()}
        final["loss"] = running_loss / max(1, total)

        logger.info("Test Results:")
        for k, v in final.items():
            logger.info(f"{k}: {v:.4f}")

        return final

    def predict(
        self, image: bytes, top_k: int | None = None
    ) -> list[SimilarityPrediction]:
        """
        Возвращает top-k [{label, prob}], отсортированные по убыванию вероятности.
        """
        if not self.loaded:
            self.load()

        # маппинг классов
        if self.id2label is None:
            self.id2label = _load_id2label(self._classes_file)
            if self.id2label is None:
                raise ValueError(
                    "id2label mapping not found. Train the model at least once to generate classes.json"
                )

        self._model.eval()
        k = top_k or self.top_k
        try:
            with Image.open(io.BytesIO(image)) as img:
                img_t = self._transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(img_t)
                probs = F.softmax(logits, dim=1)

            values, indices = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
            values = values.squeeze(0).cpu().tolist()
            indices = indices.squeeze(0).cpu().tolist()

            return [
                SimilarityPrediction(name=self.id2label[idx], confidence=float(p))
                for idx, p in zip(indices, values)
            ]
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}") from e


celebrity_matcher = CelebrityMatcherModel()
