from pydantic import Field

from .base import Base


class Celebrity(Base):
    name: str = Field(examples=["Jeffrey Epstein"], description="Celebrity name")
    img_path: str = Field(examples=["jeffrey_epstein.jpg"], description="Image path")
