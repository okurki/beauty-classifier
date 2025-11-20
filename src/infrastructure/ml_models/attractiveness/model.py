import io
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from ..base_model import ModelBase

try:
    import mlflow
    import torchmetrics
    from tqdm import tqdm

    from .dataset import get_data_loaders
except ImportError:
    pass

logger = logging.getLogger(__name__)


class AttractivenessModel(ModelBase):
    _name: str = "attractiveness_classifier"

    def __init__(self, out_features: int = 512):
        super().__init__()
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        model = models.resnet50(weights="IMAGENET1K_V2")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, 1),
            nn.Sigmoid(),
        )
        self._model = model
        self._model.to(self._device)

    def _setup_metrics(self, device):
        return nn.ModuleDict(
            {
                "mse": torchmetrics.MeanSquaredError().to(device),
                "explained_variance": torchmetrics.ExplainedVariance().to(device),
            }
        )

    def _setup_correlation_metrics(self, device):
        """Metrics that need full dataset (correlation-based)"""
        return {
            "pearson": torchmetrics.PearsonCorrCoef().to(device),
            "spearman": torchmetrics.SpearmanCorrCoef().to(device),
        }

    def train(
        self,
        epochs: int = 10,
        lr: float = 0.001,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        if not self.loaded:
            self.load()

        if optimizer is None:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )

        train_loader, val_loader = get_data_loaders()

        metrics = self._setup_metrics(self._device)
        best_loss = float("inf")

        prev_epoch_loss: float = 0.0

        mlflow.set_experiment("Training AttractivenessClassifier model")

        mlflow.start_run(
            run_name=f"Attractiveness {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": scheduler.__class__.__name__ if scheduler else "None",
                "model_name": "resnet50",
                "out_features": 512,
                "fine_tuning": "feature_extraction",
            }
        )

        # Log model architecture as tag
        mlflow.set_tag("model_architecture", "resnet50_frozen_fc_custom")

        mlflow.log_artifact(__file__, artifact_path="code")

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            for phase, dataloader in [("train", train_loader), ("val", val_loader)]:
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()

                # Reset ALL metrics at phase start
                for metric in metrics.values():
                    metric.reset()

                running_loss = 0.0
                total_samples = 0

                progress_bar = tqdm(
                    dataloader, desc=f"{phase.capitalize()} Epoch {epoch + 1}"
                )

                for batch_idx, (inputs, labels) in enumerate(progress_bar):
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device).view(-1, 1)
                    batch_size = inputs.size(0)
                    total_samples += batch_size

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self._model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * batch_size

                    outputs_squeezed = outputs.squeeze(-1)
                    labels_squeezed = labels.squeeze(-1)

                    for name, metric in metrics.items():
                        metric.update(outputs_squeezed, labels_squeezed)

                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "mse": f"{metrics['mse'].compute().item():.4f}",
                        }
                    )

                epoch_loss = running_loss / total_samples  # Weighted average

                computed_metrics = {}
                for name, metric in metrics.items():
                    computed_metrics[name] = metric.compute().item()

                mse_diff = epoch_loss - prev_epoch_loss

                mlflow.log_metrics(
                    {
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_mse": computed_metrics["mse"],
                        f"{phase}_explained_variance": computed_metrics[
                            "explained_variance"
                        ],
                        f"{phase}_mse_diff": mse_diff,
                    },
                    step=epoch,
                )

                logger.info(f"{phase.capitalize()} Results:")
                logger.info(f"Previous MSE: {prev_epoch_loss:.6f}")
                logger.info(f"Current MSE:    {epoch_loss:.6f}")
                logger.info(f"Difference:    {mse_diff:.6f}")

                for name, value in computed_metrics.items():
                    if name != "mse":  # Skip MSE since we already logged it
                        logger.info(f"  {name.upper()}: {value:.4f}")

                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save()
                    mlflow.pytorch.log_model(
                        self._model,
                        "best_model",
                        registered_model_name="attractiveness_classifier",
                    )
                    mlflow.log_metric("best_val_loss", best_loss)
                    logger.info(f"New best model saved with val loss: {best_loss:.4f}")

                prev_epoch_loss = epoch_loss

            # Scheduler step using validation criterion loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

        mlflow.log_metric("final_best_val_loss", best_loss)

        mlflow.set_tag("best_model", "True")

        mlflow.end_run()

        logger.info(f"Training complete. Best val loss: {best_loss:.4f}")

    def evaluate(self) -> dict:
        if not self.loaded:
            self.load()

        self._model.eval()

        # Batch metrics
        batch_metrics = self._setup_metrics(self._device)

        # Correlation metrics (need full dataset)
        corr_metrics = self._setup_correlation_metrics(self._device)

        _, test_loader = get_data_loaders()
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device).view(-1, 1)

                outputs = self._model(inputs)

                # Store for correlation metrics
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                # Update batch metrics
                for name, metric in batch_metrics.items():
                    metric.update(outputs.squeeze(), labels.squeeze())

                # Update progress bar
                current_metrics = {}
                for name, metric in batch_metrics.items():
                    try:
                        val = metric.compute().item()
                        current_metrics[name] = f"{val:.4f}"
                    except AttributeError:
                        pass
                progress_bar.set_postfix(current_metrics)

        # Compute final batch metrics
        final_metrics = {
            name: metric.compute().item() for name, metric in batch_metrics.items()
        }

        # Compute correlation metrics on full dataset
        all_predictions = torch.tensor(all_predictions).to(self._device)
        all_labels = torch.tensor(all_labels).to(self._device)

        for name, metric in corr_metrics.items():
            metric.update(all_predictions, all_labels)
            final_metrics[name] = metric.compute().item()

        # Log results
        logger.info("Test Results:")
        for metric, value in final_metrics.items():
            logger.info(f"Test {metric.upper()}: {value:.4f}")

        return final_metrics

    def predict(self, image: bytes) -> float:
        """Predict face attractiveness from image"""
        self._model.eval()
        try:
            with Image.open(io.BytesIO(image)) as image:
                image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = self._model(image_tensor)

            return output.item()
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}") from e


attractiveness_model = AttractivenessModel()
