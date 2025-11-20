from .crud import CRUDRepository
from ..database.models import Inference


class InferenceRepository(CRUDRepository[Inference]):
    pass
