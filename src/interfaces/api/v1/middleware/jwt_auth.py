import logging

from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.application.services import SecurityService
from src.interfaces.api.v1.schemas.token import Token

from .exc import UnauthorizedException, ForbiddenException

logger = logging.getLogger(__name__)


class JWTAuth(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTAuth, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTAuth, self).__call__(
            request
        )
        if not credentials or credentials.scheme != "Bearer":
            raise UnauthorizedException

        token = SecurityService.decode_token(credentials.credentials)
        if not token:
            logger.debug("Unable to decode token")
            raise UnauthorizedException

        await self._check_permissions(request, token)
        return token

    async def _check_permissions(self, request: Request, token: Token) -> None:
        if token.role == "admin":
            return  # allow admin access

        if "users/me" in request.url.path:
            return  # allow access to own resources

        requested_id = request.path_params.get("id") or request.path_params.get(
            "user_id"
        )

        if requested_id and str(token.user_id) == str(requested_id):
            return  # allow access to own resources

        logger.debug(
            f"Access denied. User <{token.user_id}> tried to access <{requested_id}>"
        )
        raise ForbiddenException(
            requested=f"user/{requested_id}", available=f"user/{token.user_id}"
        )
