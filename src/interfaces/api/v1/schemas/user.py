from datetime import datetime

from pydantic import Field

from .base import Base, IDMixin
from .celebrity import Celebrity


class Inference(Base, IDMixin):
    user_id: int = Field(description="User ID")
    celebrities: list[Celebrity] = Field(description="List of celebrities")
    attractiveness: float = Field(examples=[4.0], description="Attractiveness score")
    timestamp: datetime = Field(examples=[datetime.now()], description="Inference date")


class UserBase(Base):
    name: str = Field(
        examples=["name"], min_length=3, max_length=30, description="User name"
    )


class UserCreate(UserBase):
    password: str = Field(
        examples=["password"], min_length=3, max_length=30, description="User password"
    )


class UserUpdate(UserBase):
    pass


class UserRead(IDMixin, UserBase):
    inferences: list[Inference] = Field([], description="List of inferences")
