from fastapi import HTTPException, status


class UnauthorizedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization. Missing or incorrect credentials scheme",
        )


class ForbiddenException(HTTPException):
    def __init__(self, requested: str, available: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Insufficient permissions. "
                f"Tried to access <{requested}>. Available: <{available}>"
            ),
        )
