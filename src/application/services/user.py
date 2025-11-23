from typing import override
import logging


from src.infrastructure.repositories import UserRepository
from src.interfaces.api.v1.schemas import UserCreate, UserRead, InferenceRead
from .crud import CRUDService
from .security import SecurityService

logger = logging.getLogger(__name__)


class UserService(CRUDService[UserRepository, UserRead]):
    @override
    async def create(self, **data):
        data["password"] = SecurityService.hash_password(data["password"])
        user = await self.repository.create(**data)
        if not user:
            return None
        return UserRead.model_validate(user)

    @override
    async def get(self, id_login: int | str):
        user = await self.repository.get(id_login)
        if not user:
            return None
        return UserRead.model_validate(user)

    async def get_inferences(self, id: int):
        return [
            InferenceRead.model_validate(inference)
            for inference in await self.repository.get_inferences(id)
        ]

    async def login(self, data: UserCreate):
        user = await self.repository.get(data.name)

        if not user or not SecurityService.verify_password(
            data.password, user.password
        ):
            return None  # not exists or incorrect password

        return SecurityService.issue_token(
            user_id=user.id, username=user.name, role=user.role
        )

    async def register(self, data: UserCreate):
        user = await self.repository.get(data.name)
        if user:
            return None  # already exists

        data.password = SecurityService.hash_password(data.password)
        user = await self.repository.create(**data.model_dump())
        if not user:
            return None
        return SecurityService.issue_token(
            user_id=user.id, username=user.name, role=user.role
        )
