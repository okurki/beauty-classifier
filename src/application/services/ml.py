import datetime
import logging

from src.infrastructure.ml_models import attractiveness_model, celebrity_matcher
from src.interfaces.api.v1.schemas import Inference, SimilarityPrediction, Celebrity
from src.infrastructure.repositories import InferenceRepository

from .crud import CRUDService

logger = logging.getLogger(__name__)


class MLService(CRUDService[InferenceRepository, Inference]):
    def get_attractiveness(self, image: bytes) -> float | None:
        if not image:
            logger.debug("No image provided")
            return None
        if not attractiveness_model.loaded:
            logger.debug("Attractiveness model not loaded")
            attractiveness_model.load()
        try:
            predicted_attractiveness = attractiveness_model.predict(image)
        except ValueError:
            logger.debug("Could not process image")
            return None
        return round(predicted_attractiveness, 4)

    def predict_сelebrities(
        self, image: bytes, top_k: int | None = None
    ) -> list[SimilarityPrediction] | None:
        if not image:
            logger.debug("No image provided")
            return None
        if not celebrity_matcher.loaded:
            logger.debug("Celebrity matcher not loaded")
            celebrity_matcher.load()
        try:
            predictions = celebrity_matcher.predict(image, top_k)
        except ValueError:
            logger.debug("Could not process image")
            return None
        return predictions

    def create_inference(
        self, user_id: int, image: bytes, top_k: int | None = None
    ) -> Inference | None:
        celebrities = self.predict_сelebrities(image, top_k)
        if celebrities is None:
            return None
        attractiveness = self.get_attractiveness(image)
        if attractiveness is None:
            return None
        celebrities = [
            Celebrity(
                name=" ".join(
                    [name_part.capitalize() for name_part in celeb.name.split("_")]
                ),
                img_path=f"/celebrities/{celeb.name}",
            )
            for celeb in celebrities
        ]
        return Inference(
            id=1,
            user_id=user_id,
            celebrities=celebrities,
            attractiveness=attractiveness,
            timestamp=datetime.datetime.now(),
        )
