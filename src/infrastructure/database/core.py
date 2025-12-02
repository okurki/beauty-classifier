import subprocess
import logging
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

logger = logging.getLogger(__name__)


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
        cls.run_migrations()
        cls.engine = create_async_engine(
            config.db.uri,
            pool_size=config.db.pool_size,
            pool_timeout=config.db.pool_timeout,
            pool_pre_ping=True,
        )
        yield cls
        await cls.engine.dispose()

    @classmethod
    def run_migrations(cls):
        try:
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Migrations applied successfully: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Migration failed: {e.stderr}")
            logger.error(f"stdout: {e.stdout}")
            raise

    @classmethod
    async def session(cls) -> AsyncGenerator[AsyncSession]:
        async with cls.async_sessionmaker() as session:
            async with session.begin():
                yield session
