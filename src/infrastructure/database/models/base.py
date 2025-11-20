from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CreatedAtMixin(Base):
    __abstract__ = True
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, index=True, server_default=func.now()
    )


class EntityBase(CreatedAtMixin):
    __abstract__ = True
    id: Mapped[int] = mapped_column(primary_key=True)
