from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, HTTPException, status

from src.interfaces.api.v1.middleware import JWTAuth
from src.interfaces.api.v1.schemas import Token, InferenceRead, FeedbackRequest, FeedbackRead
from ..schemas import AttractivenessPrediction, SimilarityPrediction
from src.application.services.ml import MLService
from src.application.services.feedback import FeedbackService
from src.infrastructure.ml_models.rl.agent import rl_agent
from src.interfaces.api.v1.middleware.prometheus import update_rl_metrics

ml_router = APIRouter(prefix="/ml", tags=["ML"])


@ml_router.post("/attractiveness")
async def predict(
    image_file: UploadFile,
    service: Annotated[MLService, Depends(MLService.dep())],
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
    service: Annotated[MLService, Depends(MLService.dep())],
) -> list[SimilarityPrediction]:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    celebrities = await service.get_celebrities(image_bytes)

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
    service: Annotated[MLService, Depends(MLService.dep())],
) -> InferenceRead:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    inference = await service.create_inference(token.user_id, image_bytes)

    if not inference:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process image",
        )

    return inference


@ml_router.post("/users/me/feedback")
async def create_feedback(
    feedback_request: FeedbackRequest,
    token: Annotated[Token, Depends(JWTAuth())],
    service: Annotated[FeedbackService, Depends(FeedbackService.dep())],
) -> FeedbackRead:
    """
    Submit feedback (like/dislike) for a celebrity match in an inference.
    This feedback will be used to improve future predictions via RL agent.
    """
    feedback = await service.create_feedback(token.user_id, feedback_request)

    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create feedback",
        )

    # Update Prometheus metrics
    update_rl_metrics()

    return feedback


@ml_router.get("/users/me/feedback")
async def get_my_feedbacks(
    token: Annotated[Token, Depends(JWTAuth())],
    service: Annotated[FeedbackService, Depends(FeedbackService.dep())],
) -> list[FeedbackRead]:
    """
    Get all feedbacks submitted by the current user.
    """
    return await service.get_user_feedbacks(token.user_id)


@ml_router.get("/users/me/feedback/stats")
async def get_my_feedback_stats(
    token: Annotated[Token, Depends(JWTAuth())],
    service: Annotated[FeedbackService, Depends(FeedbackService.dep())],
) -> dict:
    """
    Get statistics about the current user's feedback activity.
    """
    return await service.get_user_feedback_stats(token.user_id)


@ml_router.get("/celebrities/{celebrity_id}/feedback/stats")
async def get_celebrity_feedback_stats(
    celebrity_id: int,
    service: Annotated[FeedbackService, Depends(FeedbackService.dep())],
) -> dict:
    """
    Get feedback statistics for a specific celebrity.
    Shows total likes, dislikes, and overall score.
    """
    return await service.get_celebrity_feedback_stats(celebrity_id)


@ml_router.post("/admin/reload-feedback-weights")
async def reload_feedback_weights(
    token: Annotated[Token, Depends(JWTAuth())],
    ml_service: Annotated[MLService, Depends(MLService.dep())],
) -> dict:
    """
    Reload all feedback weights from database into the model.
    This endpoint is useful for syncing model state after restart or bulk feedback updates.
    Requires authentication.
    """
    await ml_service.load_feedback_weights()
    return {"status": "success", "message": "Feedback weights reloaded successfully"}


# ==================== RL Agent Endpoints ====================

@ml_router.get("/rl/stats")
async def get_rl_stats() -> dict:
    """
    Get global RL agent statistics.
    
    Returns metrics like cumulative reward, exploration rate, CTR, etc.
    """
    return rl_agent.get_global_stats()


@ml_router.get("/rl/celebrity/{celebrity_id}")
async def get_rl_celebrity_stats(celebrity_id: int) -> dict:
    """
    Get RL statistics for a specific celebrity.
    
    Shows Beta distribution parameters (alpha, beta), mean probability,
    uncertainty, and impression counts.
    """
    stats = rl_agent.get_celebrity_stats(celebrity_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Celebrity {celebrity_id} not found in RL agent",
        )
    return stats


@ml_router.get("/rl/top-celebrities")
async def get_rl_top_celebrities(top_k: int = 10) -> list[dict]:
    """
    Get top-K celebrities by RL agent's mean probability.
    
    This shows which celebrities the RL agent believes are most likely
    to receive positive feedback.
    """
    return rl_agent.get_top_celebrities(top_k)


@ml_router.post("/rl/reset")
async def reset_rl_agent(
    token: Annotated[Token, Depends(JWTAuth())],
) -> dict:
    """
    Reset RL agent to initial state (admin only).
    
    This clears all learned statistics and restarts exploration.
    Use with caution!
    """
    rl_agent.reset()
    return {"status": "success", "message": "RL agent reset successfully"}


@ml_router.get("/rl/metrics")
async def get_rl_metrics() -> dict:
    """
    Get detailed RL performance metrics.
    
    Includes:
    - Cumulative reward
    - Average reward
    - Click-through rate (CTR)
    - Like rate
    - Exploration vs exploitation ratio
    - Total impressions and feedback counts
    """
    stats = rl_agent.get_global_stats()
    
    # Add additional computed metrics
    stats["regret_estimate"] = max(0, stats["total_impressions"] - stats["cumulative_reward"])
    
    return {
        "status": "success",
        "metrics": stats,
        "description": {
            "cumulative_reward": "Total likes minus dislikes",
            "average_reward": "Mean reward per impression",
            "ctr": "Percentage of impressions that received feedback",
            "like_rate": "Percentage of feedback that was positive",
            "exploration_rate": "Percentage of exploratory actions",
            "regret_estimate": "Estimated cumulative regret (upper bound)",
        }
    }
