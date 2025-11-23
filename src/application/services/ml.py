import logging
from datetime import datetime

from src.infrastructure.ml_models import attractiveness_model, celebrity_matcher
from src.infrastructure.repositories import InferenceRepository
from src.interfaces.api.v1.schemas import InferenceCreate, InferenceRead, Celebrity

from .crud import CRUDService

logger = logging.getLogger(__name__)


class MLService(CRUDService[InferenceRepository, InferenceCreate]):
    repository: InferenceRepository

    def get_attractiveness(self, image: bytes):
        if not image:
            logger.debug("No image provided")
            return None
        if not attractiveness_model.loaded:
            logger.error("Attractiveness model not loaded")
            attractiveness_model.load()
        try:
            predicted_attractiveness = attractiveness_model.predict(image)
        except ValueError:
            logger.debug("Could not process image")
            return None
        return round(predicted_attractiveness, 4)

    def get_celebrities(self, image: bytes, top_k: int | None = None):
        if not image:
            logger.debug("No image provided")
            return None
        if not celebrity_matcher.loaded:
            logger.error("Celebrity matcher not loaded")
            celebrity_matcher.load()
        try:
            predictions = celebrity_matcher.predict(image, top_k)
        except ValueError as e:
            logger.debug(f"Could not process image: {e}")
            return None
        return [
            Celebrity(
                name=" ".join(
                    [name_part.capitalize() for name_part in celeb.name.split("_")]
                ),
                img_path=f"/celebrities_pretty/{celeb.name}.jpg",
            )
            for celeb in predictions
        ]

    async def create_inference(
        self, user_id: int, image: bytes, top_k: int | None = None
    ):
        celebrities = self.get_celebrities(image, top_k)
        if celebrities is None:
            return None
        attractiveness = self.get_attractiveness(image)
        if attractiveness is None:
            return None

        inference = InferenceCreate(
            user_id=user_id,
            celebrities=celebrities,
            attractiveness=attractiveness,
            timestamp=datetime.now(),
        )

        inference_db = await self.repository.create(inference)
        if not inference_db:
            return None

        return InferenceRead.model_validate(inference_db)
