import logging
from datetime import datetime

from sqlalchemy import select
from src.infrastructure.ml_models import attractiveness_model, celebrity_matcher
from src.infrastructure.repositories import InferenceRepository, FeedbackRepository, CelebrityRepository
from src.infrastructure.database.models.public import CelebrityFeedback, Celebrity as CelebrityModel
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

    async def load_feedback_weights(self):
        """
        Load all feedback from database and update celebrity matcher weights.
        Should be called periodically or after new feedback is submitted.
        """
        try:
            feedback_repo = FeedbackRepository(self.repository.db)
            # Get all feedbacks from database
            result = await self.repository.db.execute(
                select(CelebrityFeedback, CelebrityModel)
                .join(CelebrityModel, CelebrityFeedback.celebrity_id == CelebrityModel.id)
            )
            rows = result.all()
            
            feedbacks = []
            for feedback, celebrity in rows:
                # Convert celebrity display name back to internal format
                # "John Doe" -> "john_doe"
                celebrity_name = "_".join(celebrity.name.lower().split())
                feedbacks.append({
                    "celebrity_name": celebrity_name,
                    "feedback_type": feedback.feedback_type.value,
                })
            
            celebrity_matcher.load_feedback_from_db(feedbacks)
            logger.info(f"Loaded {len(feedbacks)} feedbacks into celebrity matcher")
        except Exception as e:
            logger.error(f"Failed to load feedback weights: {e}")

    async def get_celebrities(self, image: bytes, top_k: int | None = None, apply_feedback: bool = True):
        if not image:
            logger.debug("No image provided")
            return None
        if not celebrity_matcher.loaded:
            logger.error("Celebrity matcher not loaded")
            celebrity_matcher.load()
        try:
            predictions = celebrity_matcher.predict(image, top_k, apply_feedback=apply_feedback)
        except ValueError as e:
            logger.debug(f"Could not process image: {e}")
            return None
        
        # Get or create celebrities in database to get their IDs
        result_celebrities = []
        
        for celeb in predictions:
            # Format name for display
            display_name = " ".join(
                [name_part.capitalize() for name_part in celeb.name.split("_")]
            )
            img_path = f"/celebrities_pretty/{celeb.name}.jpg"
            
            # Check if celebrity exists in database
            query = select(CelebrityModel).where(
                CelebrityModel.name == display_name
            )
            result = await self.repository.db.execute(query)
            celeb_db = result.scalar_one_or_none()
            
            if celeb_db:
                # Celebrity exists, use their ID
                result_celebrities.append(
                    Celebrity(
                        id=celeb_db.id,
                        name=display_name,
                        img_path=img_path,
                    )
                )
            else:
                # Celebrity doesn't exist, create without ID (will be created in inference)
                result_celebrities.append(
                    Celebrity(
                        id=None,
                        name=display_name,
                        img_path=img_path,
                    )
                )
        
        return result_celebrities

    async def create_inference(
        self, user_id: int, image: bytes, top_k: int | None = None
    ):
        celebrities = await self.get_celebrities(image, top_k)
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
