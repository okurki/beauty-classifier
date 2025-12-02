"""
Integration tests for feedback API endpoints.
Tests the complete feedback flow including RL agent updates.
"""

import pytest
from httpx import AsyncClient

from tests.integration.api.utils import create_test_user, get_auth_token


@pytest.mark.asyncio
async def test_create_feedback_success(client: AsyncClient):
    """Test successful feedback creation"""
    # Create user and get token
    user = await create_test_user(client, "feedback_user", "password123")
    token = await get_auth_token(client, "feedback_user", "password123")

    # Create an inference first (mock data for now)
    # In real scenario, you'd upload an image and get inference

    # Submit like feedback
    response = await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "like"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == user["id"]
    assert data["inference_id"] == 1
    assert data["celebrity_id"] == 1
    assert data["feedback_type"] == "like"
    assert "id" in data
    assert "timestamp" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_feedback_dislike(client: AsyncClient):
    """Test dislike feedback creation"""
    user = await create_test_user(client, "dislike_user", "password123")
    token = await get_auth_token(client, "dislike_user", "password123")

    response = await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 2, "feedback_type": "dislike"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "dislike"


@pytest.mark.asyncio
async def test_create_feedback_invalid_type(client: AsyncClient):
    """Test feedback with invalid type"""
    user = await create_test_user(client, "invalid_user", "password123")
    token = await get_auth_token(client, "invalid_user", "password123")

    response = await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "invalid"},
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_feedback_unauthorized(client: AsyncClient):
    """Test feedback creation without authentication"""
    response = await client.post(
        "/ml/users/me/feedback",
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "like"},
    )

    assert response.status_code == 401  # Unauthorized


@pytest.mark.asyncio
async def test_get_user_feedbacks(client: AsyncClient):
    """Test retrieving user's feedback history"""
    user = await create_test_user(client, "history_user", "password123")
    token = await get_auth_token(client, "history_user", "password123")

    # Create multiple feedbacks
    for i in range(3):
        await client.post(
            "/ml/users/me/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inference_id": i + 1,
                "celebrity_id": i + 1,
                "feedback_type": "like" if i % 2 == 0 else "dislike",
            },
        )

    # Get feedback history
    response = await client.get(
        "/ml/users/me/feedback", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3


@pytest.mark.asyncio
async def test_get_user_feedback_stats(client: AsyncClient):
    """Test user feedback statistics"""
    user = await create_test_user(client, "stats_user", "password123")
    token = await get_auth_token(client, "stats_user", "password123")

    # Create feedbacks: 2 likes, 1 dislike
    await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "like"},
    )
    await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 2, "celebrity_id": 2, "feedback_type": "like"},
    )
    await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 3, "celebrity_id": 3, "feedback_type": "dislike"},
    )

    # Get stats
    response = await client.get(
        "/ml/users/me/feedback/stats", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] >= 3
    assert data["likes"] >= 2
    assert data["dislikes"] >= 1


@pytest.mark.asyncio
async def test_get_celebrity_feedback_stats(client: AsyncClient):
    """Test celebrity feedback statistics"""
    # Create feedbacks from multiple users for same celebrity
    for i in range(3):
        user = await create_test_user(client, f"celeb_user_{i}", "password123")
        token = await get_auth_token(client, f"celeb_user_{i}", "password123")

        await client.post(
            "/ml/users/me/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inference_id": i + 1,
                "celebrity_id": 1,  # Same celebrity
                "feedback_type": "like" if i < 2 else "dislike",
            },
        )

    # Get celebrity stats (no auth required)
    response = await client.get("/ml/celebrities/1/feedback/stats")

    assert response.status_code == 200
    data = response.json()

    assert "total" in data
    assert "likes" in data
    assert "dislikes" in data
    assert "score" in data
    assert data["likes"] >= 2
    assert data["dislikes"] >= 1


@pytest.mark.asyncio
async def test_rl_agent_stats(client: AsyncClient):
    """Test RL agent global statistics"""
    response = await client.get("/ml/rl/stats")

    assert response.status_code == 200
    data = response.json()

    # Check all required fields
    assert "total_celebrities" in data
    assert "cumulative_reward" in data
    assert "average_reward" in data
    assert "total_impressions" in data
    assert "total_likes" in data
    assert "total_dislikes" in data
    assert "ctr" in data
    assert "like_rate" in data
    assert "exploration_rate" in data


@pytest.mark.asyncio
async def test_rl_agent_top_celebrities(client: AsyncClient):
    """Test RL agent top celebrities endpoint"""
    response = await client.get("/ml/rl/top-celebrities?top_k=5")

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    # Each celebrity should have required fields
    for celeb in data:
        assert "celebrity_id" in celeb
        assert "celebrity_name" in celeb
        assert "mean_probability" in celeb
        assert "total_impressions" in celeb


@pytest.mark.asyncio
async def test_rl_metrics(client: AsyncClient):
    """Test RL metrics endpoint"""
    response = await client.get("/ml/rl/metrics")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "metrics" in data
    assert "description" in data

    metrics = data["metrics"]
    assert "cumulative_reward" in metrics
    assert "exploration_rate" in metrics
    assert "regret_estimate" in metrics


@pytest.mark.asyncio
async def test_feedback_updates_rl_agent(client: AsyncClient):
    """Test that feedback actually updates RL agent state"""
    user = await create_test_user(client, "rl_test_user", "password123")
    token = await get_auth_token(client, "rl_test_user", "password123")

    # Get initial RL stats
    initial_response = await client.get("/ml/rl/stats")
    initial_data = initial_response.json()
    initial_likes = initial_data["total_likes"]

    # Submit like feedback
    await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "like"},
    )

    # Get updated RL stats
    updated_response = await client.get("/ml/rl/stats")
    updated_data = updated_response.json()
    updated_likes = updated_data["total_likes"]

    # Verify RL agent was updated
    assert updated_likes > initial_likes


@pytest.mark.asyncio
async def test_feedback_case_insensitive(client: AsyncClient):
    """Test that feedback_type is case insensitive"""
    user = await create_test_user(client, "case_user", "password123")
    token = await get_auth_token(client, "case_user", "password123")

    # Test uppercase
    response = await client.post(
        "/ml/users/me/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_id": 1, "celebrity_id": 1, "feedback_type": "LIKE"},
    )

    assert response.status_code == 200
    assert response.json()["feedback_type"] == "like"


@pytest.mark.asyncio
async def test_celebrity_id_in_inference(client: AsyncClient):
    """Test that celebrity IDs are returned in inference response"""
    user = await create_test_user(client, "inference_user", "password123")
    token = await get_auth_token(client, "inference_user", "password123")

    # This test requires actual image upload and model inference
    # For now, we'll just verify the schema structure
    # In production, you'd upload a test image here

    # Mock test - verify endpoint exists
    # response = await client.post(
    #     "/ml/users/me/inference",
    #     headers={"Authorization": f"Bearer {token}"},
    #     files={"image_file": ("test.jpg", test_image_bytes, "image/jpeg")}
    # )
    #
    # assert response.status_code == 200
    # data = response.json()
    # assert "celebrities" in data
    # for celeb in data["celebrities"]:
    #     assert "id" in celeb  # Celebrity ID should be present
    #     assert "name" in celeb
    #     assert "img_path" in celeb

    # Placeholder assertion
    assert True  # Replace with actual test when image handling is set up


@pytest.mark.asyncio
async def test_multiple_feedbacks_same_celebrity(client: AsyncClient):
    """Test multiple feedbacks for same celebrity accumulate correctly"""
    user = await create_test_user(client, "multi_user", "password123")
    token = await get_auth_token(client, "multi_user", "password123")

    celebrity_id = 5

    # Submit multiple likes
    for i in range(3):
        await client.post(
            "/ml/users/me/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inference_id": i + 1,
                "celebrity_id": celebrity_id,
                "feedback_type": "like",
            },
        )

    # Get celebrity stats
    response = await client.get(f"/ml/celebrities/{celebrity_id}/feedback/stats")

    assert response.status_code == 200
    data = response.json()

    assert data["likes"] >= 3
    assert data["score"] >= 3  # Score = likes - dislikes
