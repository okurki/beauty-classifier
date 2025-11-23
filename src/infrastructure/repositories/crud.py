from abc import ABC
from typing import TypeVar, Annotated, get_args

from sqlalchemy import select, delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from src.infrastructure.database import DB
from src.infrastructure.database.models.base import EntityBase

ModelType = TypeVar("ModelType", bound=EntityBase)


class CRUDRepository[ModelType: EntityBase](ABC):
    model: type[ModelType]

    def __init__(self, db: Annotated[AsyncSession, Depends(DB.session)]):
        self.db = db
        self.model = get_args(self.__class__.__orig_bases__[0])[0]

    async def get(self, id: int):
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def find_many(self, offset: int, limit: int):
        query = select(self.model).offset(offset).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def create(self, **data):
        try:
            instance = self.model(**data)
            self.db.add(instance)
            await self.db.flush()
            await self.db.refresh(instance)
            return instance
        except IntegrityError:
            return None

    async def update(self, id: int, **data):
        query = update(self.model).where(self.model.id == id).values(data)
        try:
            result = await self.db.execute(query)
        except IntegrityError:
            self.db.rollback()
            return None
        return result.rowcount > 0

    async def delete(self, id: int):
        query = delete(self.model).where(self.model.id == id)
        try:
            result = await self.db.execute(query)
        except IntegrityError:
            self.db.rollback()
            return None
        return result.rowcount > 0
