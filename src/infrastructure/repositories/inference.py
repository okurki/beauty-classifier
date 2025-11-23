from typing import override

from src.interfaces.api.v1.schemas import InferenceCreate
from .crud import CRUDRepository
from ..database.models import Inference, Celebrity


class InferenceRepository(CRUDRepository[Inference]):
    @override
    async def create(self, model: InferenceCreate):
        """
        Overloaded to create from InferenceCreate model
        """
        inferece = Inference(**model.model_dump(exclude=["celebrities"]))
        celebrities = [
            Celebrity(
                name=celeb.name,
                img_path=celeb.img_path,
            )
            for celeb in model.celebrities
        ]
        inferece.celebrities = celebrities
        self.db.add(inferece)
        await self.db.flush()
        await self.db.refresh(inferece)
        return inferece
