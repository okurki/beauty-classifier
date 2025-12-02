from typing import override

from sqlalchemy import select
from src.interfaces.api.v1.schemas import InferenceCreate
from .crud import CRUDRepository
from ..database.models.public import Inference, Celebrity


class InferenceRepository(CRUDRepository[Inference]):
    @override
    async def create(self, model: InferenceCreate):
        """
        Overloaded to create from InferenceCreate model.
        Handles celebrities with existing IDs or creates new ones.
        """
        inferece = Inference(**model.model_dump(exclude=["celebrities"]))
        celebrities = []

        for celeb in model.celebrities:
            if celeb.id:
                # Celebrity has ID, fetch from database
                query = select(Celebrity).where(Celebrity.id == celeb.id)
                result = await self.db.execute(query)
                celeb_db = result.scalar_one_or_none()

                if celeb_db:
                    celebrities.append(celeb_db)
                else:
                    # ID provided but not found, create new
                    celebrities.append(
                        Celebrity(
                            name=celeb.name,
                            img_path=celeb.img_path,
                        )
                    )
            else:
                # No ID, check if exists by name
                query = select(Celebrity).where(Celebrity.name == celeb.name)
                result = await self.db.execute(query)
                celeb_db = result.scalar_one_or_none()

                if celeb_db:
                    celebrities.append(celeb_db)
                else:
                    # Create new celebrity
                    celebrities.append(
                        Celebrity(
                            name=celeb.name,
                            img_path=celeb.img_path,
                        )
                    )

        inferece.celebrities = celebrities
        self.db.add(inferece)
        await self.db.flush()
        await self.db.refresh(inferece)
        return inferece
