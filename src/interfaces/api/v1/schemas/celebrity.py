from pydantic import Field

from .base import Base


class Celebrity(Base):
    name: str = Field(examples=["Jeffrey Epstein"], description="Celebrity name")
    img_path: str = Field(description="Celebrity image path")
