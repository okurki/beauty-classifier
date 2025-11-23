from pydantic import BaseModel, ConfigDict, Field


class Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, strict=True, extra="ignore")


class IDMixin(BaseModel):
    id: int = Field(examples=[1], description="ID")
