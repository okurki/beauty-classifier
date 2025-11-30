from datetime import datetime
from enum import StrEnum
from typing import Union

from pydantic import field_validator, Field
from .base import Base


class FeedbackType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"


class FeedbackCreate(Base):
    user_id: int
    inference_id: int
    celebrity_id: int
    feedback_type: FeedbackType
    timestamp: datetime


class FeedbackRead(Base):
    id: int
    user_id: int
    inference_id: int
    celebrity_id: int
    feedback_type: FeedbackType
    timestamp: datetime
    created_at: datetime
    updated_at: datetime
    
    @field_validator("feedback_type", mode="before")
    @classmethod
    def convert_feedback_type(cls, v):
        """Convert string to FeedbackType enum"""
        if isinstance(v, str):
            try:
                return FeedbackType(v.lower())
            except ValueError:
                raise ValueError(f"feedback_type must be 'like' or 'dislike', got '{v}'")
        return v


class FeedbackRequest(Base):
    inference_id: int
    celebrity_id: int
    feedback_type: Union[str, FeedbackType] = Field(
        ..., 
        description="Feedback type: 'like' or 'dislike'",
        examples=["like", "dislike"]
    )
    
    @field_validator("feedback_type", mode="before")
    @classmethod
    def convert_to_enum(cls, v):
        """Convert string to FeedbackType enum"""
        if isinstance(v, str):
            try:
                return FeedbackType(v.lower())
            except ValueError:
                raise ValueError(f"feedback_type must be 'like' or 'dislike', got '{v}'")
        elif isinstance(v, FeedbackType):
            return v
        else:
            raise ValueError(f"Invalid feedback_type type: {type(v)}")

