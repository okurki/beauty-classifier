"""
Unit tests for RL Agent (Thompson Sampling).
Tests the core reinforcement learning logic.
"""

import pytest
import numpy as np

from src.infrastructure.ml_models.rl.agent import (
    ThompsonSamplingAgent,
    CelebrityState,
    RLAction,
    RLMetrics,
)


def test_celebrity_state_initialization():
    """Test CelebrityState initialization"""
    state = CelebrityState(
        celebrity_id=1,
        celebrity_name="test_celebrity",
        alpha=1.0,
        beta=1.0
    )
    
    assert state.celebrity_id == 1
    assert state.celebrity_name == "test_celebrity"
    assert state.alpha == 1.0
    assert state.beta == 1.0
    assert state.total_impressions == 0
    assert state.mean_probability == 0.5  # Beta(1,1) mean


def test_celebrity_state_mean_probability():
    """Test mean probability calculation"""
    state = CelebrityState(
        celebrity_id=1,
        celebrity_name="test",
        alpha=3.0,
        beta=1.0
    )
    
    # Mean of Beta(3,1) = 3/(3+1) = 0.75
    assert abs(state.mean_probability - 0.75) < 0.01


def test_celebrity_state_uncertainty():
    """Test uncertainty calculation"""
    state = CelebrityState(
        celebrity_id=1,
        celebrity_name="test",
        alpha=10.0,
        beta=10.0
    )
    
    # More data = less uncertainty
    assert state.uncertainty < 0.01


def test_celebrity_state_sampling():
    """Test sampling from Beta distribution"""
    state = CelebrityState(
        celebrity_id=1,
        celebrity_name="test",
        alpha=5.0,
        beta=2.0
    )
    
    # Sample multiple times
    samples = [state.sample_probability() for _ in range(100)]
    
    # All samples should be between 0 and 1
    assert all(0 <= s <= 1 for s in samples)
    
    # Mean should be close to expected
    assert abs(np.mean(samples) - state.mean_probability) < 0.1


def test_rl_metrics_initialization():
    """Test RLMetrics initialization"""
    metrics = RLMetrics()
    
    assert metrics.cumulative_reward == 0.0
    assert metrics.total_impressions == 0
    assert metrics.average_reward == 0.0
    assert metrics.ctr == 0.0


def test_rl_metrics_average_reward():
    """Test average reward calculation"""
    metrics = RLMetrics(
        cumulative_reward=10.0,
        total_impressions=5
    )
    
    assert metrics.average_reward == 2.0


def test_rl_metrics_ctr():
    """Test CTR calculation"""
    metrics = RLMetrics(
        total_impressions=100,
        total_likes=30,
        total_dislikes=20
    )
    
    # CTR = (likes + dislikes) / impressions
    assert metrics.ctr == 0.5


def test_rl_metrics_like_rate():
    """Test like rate calculation"""
    metrics = RLMetrics(
        total_likes=30,
        total_dislikes=10
    )
    
    # Like rate = likes / (likes + dislikes)
    assert metrics.like_rate == 0.75


def test_agent_initialization():
    """Test ThompsonSamplingAgent initialization"""
    agent = ThompsonSamplingAgent(
        alpha_prior=1.0,
        beta_prior=1.0,
        context_weight=0.3
    )
    
    assert agent.alpha_prior == 1.0
    assert agent.beta_prior == 1.0
    assert agent.context_weight == 0.3
    assert len(agent.celebrity_states) == 0


def test_agent_register_celebrity():
    """Test registering a new celebrity"""
    agent = ThompsonSamplingAgent()
    
    agent.register_celebrity(1, "test_celebrity")
    
    assert 1 in agent.celebrity_states
    assert agent.celebrity_states[1].celebrity_name == "test_celebrity"
    assert agent.celebrity_states[1].alpha == agent.alpha_prior
    assert agent.celebrity_states[1].beta == agent.beta_prior


def test_agent_update_like():
    """Test updating agent with like feedback"""
    agent = ThompsonSamplingAgent()
    agent.register_celebrity(1, "test_celebrity")
    
    initial_alpha = agent.celebrity_states[1].alpha
    
    agent.update(celebrity_id=1, reward=1.0)
    
    # Alpha should increase
    assert agent.celebrity_states[1].alpha == initial_alpha + 1.0
    assert agent.metrics.cumulative_reward == 1.0
    assert agent.metrics.total_likes == 1


def test_agent_update_dislike():
    """Test updating agent with dislike feedback"""
    agent = ThompsonSamplingAgent()
    agent.register_celebrity(1, "test_celebrity")
    
    initial_beta = agent.celebrity_states[1].beta
    
    agent.update(celebrity_id=1, reward=-1.0)
    
    # Beta should increase
    assert agent.celebrity_states[1].beta == initial_beta + 1.0
    assert agent.metrics.cumulative_reward == -1.0
    assert agent.metrics.total_dislikes == 1


def test_agent_select_celebrities_basic():
    """Test basic celebrity selection"""
    agent = ThompsonSamplingAgent()
    
    candidates = [
        (1, "celeb_1", 0.5),
        (2, "celeb_2", 0.3),
        (3, "celeb_3", 0.2),
    ]
    
    selected = agent.select_celebrities(
        candidate_celebrities=candidates,
        context=None,
        top_k=2
    )
    
    assert len(selected) == 2
    # Check structure
    for celeb_id, name, prob, is_exploration in selected:
        assert isinstance(celeb_id, int)
        assert isinstance(name, str)
        assert isinstance(prob, float)
        assert isinstance(is_exploration, bool)


def test_agent_select_celebrities_with_context():
    """Test celebrity selection with context"""
    agent = ThompsonSamplingAgent()
    
    candidates = [
        (1, "celeb_1", 0.5),
        (2, "celeb_2", 0.3),
    ]
    
    # Create dummy context (512-dim embedding)
    context = np.random.randn(512)
    
    selected = agent.select_celebrities(
        candidate_celebrities=candidates,
        context=context,
        top_k=2
    )
    
    assert len(selected) == 2


def test_agent_exploration_bonus():
    """Test exploration bonus for under-explored celebrities"""
    agent = ThompsonSamplingAgent(
        exploration_bonus=0.1,
        min_impressions=5
    )
    
    # Celebrity with no impressions
    bonus_new = agent._compute_exploration_bonus(999)
    assert bonus_new == agent.exploration_bonus
    
    # Register and give some impressions
    agent.register_celebrity(1, "test")
    agent.celebrity_states[1].total_impressions = 3
    
    bonus_partial = agent._compute_exploration_bonus(1)
    assert 0 < bonus_partial < agent.exploration_bonus
    
    # Celebrity with enough impressions
    agent.celebrity_states[1].total_impressions = 10
    bonus_full = agent._compute_exploration_bonus(1)
    assert bonus_full == 0.0


def test_agent_context_affinity():
    """Test context affinity calculation"""
    agent = ThompsonSamplingAgent(context_weight=0.3)
    
    # No context
    affinity = agent._compute_context_affinity(999, None)
    assert affinity == 1.0
    
    # Celebrity not registered
    context = np.random.randn(512)
    affinity = agent._compute_context_affinity(999, context)
    assert affinity == 1.0


def test_agent_get_celebrity_stats():
    """Test getting celebrity statistics"""
    agent = ThompsonSamplingAgent()
    agent.register_celebrity(1, "test_celebrity")
    agent.update(1, 1.0)
    agent.update(1, 1.0)
    agent.update(1, -1.0)
    
    stats = agent.get_celebrity_stats(1)
    
    assert stats is not None
    assert stats["celebrity_id"] == 1
    assert stats["celebrity_name"] == "test_celebrity"
    assert stats["alpha"] == 3.0  # 1 + 2 likes
    assert stats["beta"] == 2.0   # 1 + 1 dislike
    assert stats["total_likes"] == 2
    assert stats["total_dislikes"] == 1


def test_agent_get_global_stats():
    """Test getting global agent statistics"""
    agent = ThompsonSamplingAgent()
    agent.register_celebrity(1, "celeb_1")
    agent.register_celebrity(2, "celeb_2")
    
    agent.update(1, 1.0)
    agent.update(2, -1.0)
    
    stats = agent.get_global_stats()
    
    assert stats["total_celebrities"] == 2
    assert stats["cumulative_reward"] == 0.0  # 1 - 1
    assert stats["total_likes"] == 1
    assert stats["total_dislikes"] == 1


def test_agent_get_top_celebrities():
    """Test getting top celebrities"""
    agent = ThompsonSamplingAgent()
    
    # Create celebrities with different success rates
    agent.register_celebrity(1, "good_celeb")
    agent.update(1, 1.0)
    agent.update(1, 1.0)
    agent.update(1, 1.0)
    
    agent.register_celebrity(2, "bad_celeb")
    agent.update(2, -1.0)
    agent.update(2, -1.0)
    
    top = agent.get_top_celebrities(top_k=2)
    
    assert len(top) == 2
    # First should be the good celebrity
    assert top[0]["celebrity_id"] == 1
    assert top[0]["mean_probability"] > top[1]["mean_probability"]


def test_agent_reset():
    """Test resetting agent"""
    agent = ThompsonSamplingAgent()
    agent.register_celebrity(1, "test")
    agent.update(1, 1.0)
    
    assert len(agent.celebrity_states) > 0
    assert agent.metrics.cumulative_reward != 0
    
    agent.reset()
    
    assert len(agent.celebrity_states) == 0
    assert agent.metrics.cumulative_reward == 0.0


def test_agent_load_from_feedback():
    """Test loading agent from historical feedback"""
    agent = ThompsonSamplingAgent()
    
    feedbacks = [
        {"celebrity_id": 1, "celebrity_name": "celeb_1", "feedback_type": "like"},
        {"celebrity_id": 1, "celebrity_name": "celeb_1", "feedback_type": "like"},
        {"celebrity_id": 2, "celebrity_name": "celeb_2", "feedback_type": "dislike"},
    ]
    
    agent.load_from_feedback(feedbacks)
    
    assert len(agent.celebrity_states) == 2
    assert agent.celebrity_states[1].alpha == 3.0  # 1 + 2 likes
    assert agent.celebrity_states[2].beta == 2.0   # 1 + 1 dislike


def test_agent_decay_rate():
    """Test decay rate for old feedback"""
    agent = ThompsonSamplingAgent(decay_rate=0.5)
    agent.register_celebrity(1, "test")
    
    agent.update(1, 1.0)
    
    # With decay_rate=0.5, reward is halved
    assert agent.celebrity_states[1].alpha == 1.5  # 1 + 0.5


def test_agent_handles_unknown_celebrity():
    """Test agent handles feedback for unknown celebrity gracefully"""
    agent = ThompsonSamplingAgent()
    
    # Should not crash
    agent.update(999, 1.0)
    
    # Should return None for stats
    stats = agent.get_celebrity_stats(999)
    assert stats is None


def test_thompson_sampling_exploration():
    """Test that Thompson Sampling actually explores"""
    agent = ThompsonSamplingAgent()
    
    # Create two celebrities with same priors
    candidates = [
        (1, "celeb_1", 0.5),
        (2, "celeb_2", 0.5),
    ]
    
    # Select multiple times and track which is selected first
    first_selections = []
    for _ in range(100):
        selected = agent.select_celebrities(candidates, None, top_k=2)
        first_selections.append(selected[0][0])
    
    # Both should be selected as first sometimes (exploration)
    assert 1 in first_selections
    assert 2 in first_selections


def test_thompson_sampling_exploitation():
    """Test that Thompson Sampling exploits good celebrities"""
    agent = ThompsonSamplingAgent()
    
    # Give one celebrity much better feedback
    agent.register_celebrity(1, "good_celeb")
    for _ in range(10):
        agent.update(1, 1.0)
    
    agent.register_celebrity(2, "bad_celeb")
    for _ in range(10):
        agent.update(2, -1.0)
    
    candidates = [
        (1, "good_celeb", 0.5),
        (2, "bad_celeb", 0.5),
    ]
    
    # Select multiple times
    first_selections = []
    for _ in range(100):
        selected = agent.select_celebrities(candidates, None, top_k=2)
        first_selections.append(selected[0][0])
    
    # Good celebrity should be selected first most of the time
    good_count = sum(1 for x in first_selections if x == 1)
    assert good_count > 70  # At least 70% of the time

