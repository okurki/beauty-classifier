from enum import StrEnum
from datetime import datetime

from sqlalchemy import ForeignKey, Table, Column, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, EntityBase


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


association_table = Table(
    "inference_celebrities",
    Base.metadata,
    Column("inference_id", ForeignKey("inferences.id"), primary_key=True),
    Column("celebrity_id", ForeignKey("celebrities.id"), primary_key=True),
)


class User(EntityBase):
    __tablename__ = "users"
    name: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="role", create_constraint=True, native_enum=False),
        nullable=False,
        default=Role.USER,
    )
    inferences: Mapped[list["Inference"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Celebrity(EntityBase):
    __tablename__ = "celebrities"
    name: Mapped[str] = mapped_column(nullable=False, index=True)
    img_path: Mapped[str] = mapped_column(nullable=False)
    inferences: Mapped[list["Inference"]] = relationship(
        secondary=association_table, back_populates="celebrities", cascade="all, delete"
    )


class Inference(EntityBase):
    __tablename__ = "inferences"
    attractiveness: Mapped[float] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="inferences")
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    celebrities: Mapped[list["Celebrity"]] = relationship(
        secondary=association_table,
        back_populates="inferences",
        cascade="all, delete",
        lazy="selectin",
    )
    feedbacks: Mapped[list["CelebrityFeedback"]] = relationship(
        back_populates="inference", cascade="all, delete-orphan", lazy="selectin"
    )


class FeedbackType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"


class CelebrityFeedback(EntityBase):
    __tablename__ = "celebrity_feedbacks"
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    inference_id: Mapped[int] = mapped_column(
        ForeignKey("inferences.id"), nullable=False
    )
    celebrity_id: Mapped[int] = mapped_column(
        ForeignKey("celebrities.id"), nullable=False, index=True
    )
    feedback_type: Mapped[FeedbackType] = mapped_column(
        Enum(
            FeedbackType,
            name="feedback_type",
            create_constraint=True,
            native_enum=False,
        ),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False)

    user: Mapped["User"] = relationship()
    inference: Mapped["Inference"] = relationship(back_populates="feedbacks")
    celebrity: Mapped["Celebrity"] = relationship()
