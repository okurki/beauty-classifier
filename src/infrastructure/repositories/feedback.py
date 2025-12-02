from typing import override

from sqlalchemy import select

from src.interfaces.api.v1.schemas import FeedbackCreate
from .crud import CRUDRepository
from ..database.models.public import CelebrityFeedback


class FeedbackRepository(CRUDRepository[CelebrityFeedback]):
    @override
    async def create(self, model: FeedbackCreate):
        """
        Create feedback from FeedbackCreate model
        """
        feedback = CelebrityFeedback(**model.model_dump())
        self.db.add(feedback)
        await self.db.flush()
        await self.db.refresh(feedback)
        return feedback

    async def get_user_feedbacks(self, user_id: int) -> list[CelebrityFeedback]:
        """
        Get all feedbacks for a specific user
        """
        result = await self.db.execute(
            select(CelebrityFeedback).where(CelebrityFeedback.user_id == user_id)
        )
        return list(result.scalars().all())

    async def get_celebrity_feedbacks(
        self, celebrity_id: int
    ) -> list[CelebrityFeedback]:
        """
        Get all feedbacks for a specific celebrity
        """
        result = await self.db.execute(
            select(CelebrityFeedback).where(
                CelebrityFeedback.celebrity_id == celebrity_id
            )
        )
        return list(result.scalars().all())

    async def get_inference_feedbacks(
        self, inference_id: int
    ) -> list[CelebrityFeedback]:
        """
        Get all feedbacks for a specific inference
        """
        result = await self.db.execute(
            select(CelebrityFeedback).where(
                CelebrityFeedback.inference_id == inference_id
            )
        )
        return list(result.scalars().all())
