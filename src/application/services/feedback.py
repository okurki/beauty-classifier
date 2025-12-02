import logging
from datetime import datetime
from typing import Dict, List

from src.infrastructure.repositories import FeedbackRepository, CelebrityRepository
from src.interfaces.api.v1.schemas import (
    FeedbackCreate,
    FeedbackRead,
    FeedbackRequest,
    FeedbackType,
)
from src.infrastructure.ml_models import celebrity_matcher
from src.infrastructure.ml_models.rl.agent import rl_agent

from .crud import CRUDService

logger = logging.getLogger(__name__)


class FeedbackService(CRUDService[FeedbackRepository, FeedbackCreate]):
    repository: FeedbackRepository

    async def create_feedback(
        self, user_id: int, feedback_request: FeedbackRequest
    ) -> FeedbackRead | None:
        """
        Create a new feedback for a celebrity match and update model weights
        """
        feedback = FeedbackCreate(
            user_id=user_id,
            inference_id=feedback_request.inference_id,
            celebrity_id=feedback_request.celebrity_id,
            feedback_type=feedback_request.feedback_type,
            timestamp=datetime.now(),
        )

        feedback_db = await self.repository.create(feedback)
        if not feedback_db:
            logger.error("Failed to create feedback")
            return None

        logger.info(
            f"User {user_id} gave {feedback_request.feedback_type} feedback "
            f"for celebrity {feedback_request.celebrity_id} in inference {feedback_request.inference_id}"
        )

        # Update model weights and RL agent with the new feedback
        try:
            celebrity_repo = CelebrityRepository(self.repository.db)
            celebrity = await celebrity_repo.get(feedback_request.celebrity_id)

            if celebrity:
                # Convert display name back to internal format
                celebrity_name = "_".join(celebrity.name.lower().split())
                feedback_score = (
                    1.0 if feedback_request.feedback_type == FeedbackType.LIKE else -1.0
                )

                # Update simple feedback weights
                celebrity_matcher.update_feedback_weights(
                    celebrity_name, feedback_score
                )

                # Update RL agent (Thompson Sampling)
                rl_agent.update(
                    celebrity_id=feedback_request.celebrity_id,
                    reward=feedback_score,
                )

                logger.info(
                    f"Updated model weights and RL agent for celebrity {celebrity_name} "
                    f"(reward={feedback_score})"
                )
            else:
                logger.warning(f"Celebrity {feedback_request.celebrity_id} not found")
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")

        return FeedbackRead.model_validate(feedback_db)

    async def get_user_feedbacks(self, user_id: int) -> List[FeedbackRead]:
        """
        Get all feedbacks from a specific user
        """
        feedbacks = await self.repository.get_user_feedbacks(user_id)
        return [FeedbackRead.model_validate(f) for f in feedbacks]

    async def get_celebrity_feedbacks(self, celebrity_id: int) -> List[FeedbackRead]:
        """
        Get all feedbacks for a specific celebrity
        """
        feedbacks = await self.repository.get_celebrity_feedbacks(celebrity_id)
        return [FeedbackRead.model_validate(f) for f in feedbacks]

    async def get_inference_feedbacks(self, inference_id: int) -> List[FeedbackRead]:
        """
        Get all feedbacks for a specific inference
        """
        feedbacks = await self.repository.get_inference_feedbacks(inference_id)
        return [FeedbackRead.model_validate(f) for f in feedbacks]

    async def get_celebrity_feedback_stats(self, celebrity_id: int) -> Dict[str, int]:
        """
        Get statistics about feedbacks for a celebrity
        """
        feedbacks = await self.repository.get_celebrity_feedbacks(celebrity_id)

        likes = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.LIKE)
        dislikes = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.DISLIKE)

        return {
            "total": len(feedbacks),
            "likes": likes,
            "dislikes": dislikes,
            "score": likes - dislikes,
        }

    async def get_user_feedback_stats(self, user_id: int) -> Dict[str, int]:
        """
        Get statistics about user's feedback activity
        """
        feedbacks = await self.repository.get_user_feedbacks(user_id)

        likes = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.LIKE)
        dislikes = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.DISLIKE)

        return {
            "total": len(feedbacks),
            "likes": likes,
            "dislikes": dislikes,
        }
