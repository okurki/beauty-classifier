from abc import ABC
from typing import TypeVar, Generic, get_args
import logging

from pydantic import BaseModel
from fastapi import Depends

from src.infrastructure.repositories import CRUDRepository

logger = logging.getLogger(__name__)

RepositoryT = TypeVar("RepositoryT", bound=CRUDRepository)
ReadSchemaT = TypeVar("ReadSchemaT", bound=BaseModel)


class CRUDService(Generic[RepositoryT, ReadSchemaT], ABC):
    repository: RepositoryT

    def __init__(self, repository: RepositoryT, read_schema: type[ReadSchemaT]):
        self.repository = repository
        self.read_schema = read_schema

    async def create(self, **data):
        db_model = await self.repository.create(**data)
        if not db_model:
            return None
        return self.read_schema.model_validate(db_model)

    async def get(self, id: int):
        db_model = await self.repository.get(id)
        if not db_model:
            return None
        return self.read_schema.model_validate(db_model)

    async def find_many(self, page: int = 1, limit: int = 100):
        offset = (page - 1) * limit
        return [
            self.read_schema.model_validate(user)
            for user in await self.repository.find_many(offset, limit)
        ]

    async def delete(self, id: int):
        return await self.repository.delete(id)

    async def update(self, id: int, **data):
        return await self.repository.update(id, **data)

    @classmethod
    def dep(cls):
        repo = get_args(cls.__orig_bases__[0])[0]
        schema = get_args(cls.__orig_bases__[0])[1]

        def service_factory(reop=Depends(repo)):
            return cls(reop, schema)

        return service_factory
