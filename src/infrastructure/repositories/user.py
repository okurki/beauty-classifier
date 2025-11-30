from typing import override
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.public import User, Inference
from .crud import CRUDRepository


class UserRepository(CRUDRepository[User]):
    @override
    async def get(self, id_login: str | int) -> User | None:
        if isinstance(id_login, int):
            column = User.id
        elif isinstance(id_login, str):
            column = User.name
        query = (
            select(User)
            .options(selectinload(User.inferences))
            .where(column == id_login)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_inferences(self, id: int) -> list[Inference]:
        query = (
            select(Inference)
            .where(Inference.user_id == id)
            .options(selectinload(Inference.celebrities))
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_photo(self, id: int) -> bytes | None:
        query = (
            select(Inference)
            .where(Inference.user_id == id)
            .order_by(Inference.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(query)
        last_inference = result.scalar_one_or_none()
        if last_inference:
            return last_inference.picture
        return None

    async def delete_inference(self, id: int) -> bool:
        query = delete(Inference).where(Inference.id == id)
        result = await self.db.execute(query)
        return result.rowcount > 0
