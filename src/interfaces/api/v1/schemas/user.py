from datetime import datetime

from pydantic import Field

from .base import Base, IDMixin
from .celebrity import Celebrity


class InferenceCreate(Base):
    user_id: int = Field(examples=[1], description="User ID")
    celebrities: list[Celebrity] = Field(description="List of celebrities")
    attractiveness: float = Field(examples=[0.5], description="Attractiveness score")
    timestamp: datetime = Field(examples=[datetime.now()], description="Inference date")


class InferenceRead(InferenceCreate, IDMixin):
    pass


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
    inferences: list[InferenceRead] = Field(
        [],
        description="List of inferences",
        examples=[
            [
                InferenceRead(
                    id=1,
                    user_id=1,
                    celebrities=[
                        Celebrity(name="name", img_path="/celebrities_pretty/name.jpg")
                    ],
                    attractiveness=0.5,
                    timestamp=datetime.now(),
                )
            ]
        ],
    )
