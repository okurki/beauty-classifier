from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, HTTPException, status

from src.interfaces.api.v1.middleware import JWTAuth
from src.interfaces.api.v1.schemas import Token, Inference
from ..schemas import AttractivenessPrediction, SimilarityPrediction
from src.application.services.ml import MLService

ml_router = APIRouter(prefix="/ml", tags=["ML"])


@ml_router.post("/attractiveness")
async def predict(
    image_file: UploadFile,
    service: MLService = Depends(MLService.dep()),
) -> AttractivenessPrediction:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    score = service.get_attractiveness(image_bytes)

    if not score:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process image",
        )

    return AttractivenessPrediction(prediction=score)


@ml_router.post("/celebrities")
async def get_celebrities(
    image_file: UploadFile,
    service: MLService = Depends(MLService.dep()),
) -> list[SimilarityPrediction]:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    celebrities = service.predict_celebrities(image_bytes)

    if not celebrities:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process image",
        )

    return celebrities


@ml_router.post("/users/me/inference")
async def create_inference(
    image_file: UploadFile,
    token: Annotated[Token, Depends(JWTAuth())],
    service: MLService = Depends(MLService.dep()),
) -> Inference:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    inference = service.create_inference(token.user_id, image_bytes)

    if not inference:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process image",
        )

    return inference
