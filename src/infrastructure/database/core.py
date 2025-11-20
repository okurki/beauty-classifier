from typing import AsyncGenerator, ClassVar
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)

from src.config import config
from src.utils.decorators import classproperty


@dataclass(slots=True)
class DB:
    engine: ClassVar[AsyncEngine | None] = None
    _async_sessionmaker: ClassVar[async_sessionmaker | None] = None

    @classproperty
    def async_sessionmaker(cls):
        if cls.engine is None:
            raise RuntimeError("DB engine not initialized")
        if cls._async_sessionmaker is None:
            cls._async_sessionmaker = async_sessionmaker(
                cls.engine, expire_on_commit=False
            )
        return cls._async_sessionmaker

    @classmethod
    @asynccontextmanager
    async def lifespan(cls):
        cls.engine = create_async_engine(
            config.db.uri,
            pool_size=config.db.pool_size,
            pool_timeout=config.db.pool_timeout,
            pool_pre_ping=True,
        )
        yield cls
        await cls.engine.dispose()

    @classmethod
    async def session(cls) -> AsyncGenerator[AsyncSession]:
        async with cls.async_sessionmaker() as session:
            async with session.begin():
                yield session
