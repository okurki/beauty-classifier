from datetime import datetime, timezone
import logging

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import ValidationError

from src.config import config
from src.interfaces.api.v1.schemas import Token

logger = logging.getLogger(__name__)


class SecurityService:
    _password_hash = PasswordHash.recommended()

    @classmethod
    def hash_password(cls, password: str) -> str:
        return cls._password_hash.hash(password, salt=config.auth.salt)

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        return cls._password_hash.verify(plain_password, hashed_password)

    @classmethod
    def issue_token(cls, **data) -> Token:
        token_data = data.copy()

        now = int(datetime.now(timezone.utc).timestamp())
        expire_delta = config.auth.access_token_expire_m * 60
        token_data.update({"iat": now, "exp": now + expire_delta, "type": "Bearer"})

        jwt_token = jwt.encode(
            token_data, config.auth.secret_key, config.auth.algorithm
        )

        token_data["token"] = jwt_token
        return Token(**token_data)

    @classmethod
    def decode_token(cls, raw_token: str) -> Token | None:
        try:
            jwt_token: dict = jwt.decode(
                raw_token,
                config.auth.secret_key,
                algorithms=[config.auth.algorithm],
                options={"verify_exp": True},
            )
            return Token(**jwt_token, token=raw_token)
        except (InvalidTokenError, ValidationError) as e:
            if isinstance(e, ValidationError):
                logger.debug(f"Unable to coerce to Token model: {jwt_token}")
            elif isinstance(e, InvalidTokenError):
                logger.debug(f"Invalid token: {raw_token}")
            return None
