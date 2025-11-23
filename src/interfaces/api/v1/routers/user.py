from typing import Annotated

from fastapi import APIRouter, status, Depends, Query

from src.application.services import UserService
from src.interfaces.api.v1.schemas import (
    UserCreate,
    UserUpdate,
    UserRead,
    Token,
)
from src.interfaces.api.v1.exc import (
    NotFoundHTTPException,
    ObjInUseHTTPException,
    InvalidDataHTTPException,
    AlreadyExistsHTTPException,
)
from ..middleware.jwt_auth import JWTAuth

user_router = APIRouter(
    prefix="/users", tags=["User"], dependencies=[Depends(JWTAuth())]
)


@user_router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    service: Annotated[UserService, Depends(UserService.dep())],
) -> UserRead:
    id = await service.create(**data.model_dump())
    if not id:
        raise AlreadyExistsHTTPException
    return id


@user_router.get("/")
async def get_users(
    service: Annotated[UserService, Depends(UserService.dep())],
    page: int = Query(1, description="Page number"),
    limit: int = Query(100, description="Items per page"),
) -> list[UserRead]:
    return await service.find_many(page, limit)


@user_router.get("/{id}")
async def get_user(
    id: int,
    service: Annotated[UserService, Depends(UserService.dep())],
) -> UserRead:
    user = await service.get(id)
    if not user:
        raise NotFoundHTTPException
    return user


@user_router.patch("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_user(
    id: int,
    user: UserUpdate,
    service: Annotated[UserService, Depends(UserService.dep())],
) -> None:
    res = await service.update(id, **user.model_dump())
    if res is None:
        raise NotFoundHTTPException
    if res is False:
        raise InvalidDataHTTPException


@user_router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    id: int,
    service: Annotated[UserService, Depends(UserService.dep())],
) -> None:
    deleted = await service.delete(id)
    if deleted is False:
        raise NotFoundHTTPException
    if deleted is None:
        raise ObjInUseHTTPException


@user_router.get("/me/")
async def me(
    token: Annotated[Token, Depends(JWTAuth())],
    service: Annotated[UserService, Depends(UserService.dep())],
) -> UserRead:
    user = await service.get(token.user_id)
    if not user:
        raise NotFoundHTTPException
    return user
