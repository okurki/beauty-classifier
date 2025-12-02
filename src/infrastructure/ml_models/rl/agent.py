import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CelebrityState:
    """
    Represents the RL state for a single celebrity (arm).
    Uses Beta distribution to model success probability.
    """

    celebrity_id: int
    celebrity_name: str
    alpha: float = 1.0  # Prior successes (likes)
    beta: float = 1.0  # Prior failures (dislikes)
    total_impressions: int = 0
    total_likes: int = 0
    total_dislikes: int = 0
    last_updated: Optional[datetime] = None
    context_embeddings: List[np.ndarray] = field(default_factory=list)

    @property
    def mean_probability(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample_probability(self) -> float:
        return np.random.beta(self.alpha, self.beta)


@dataclass
class RLAction:
    celebrity_id: int
    celebrity_name: str
    sampled_probability: float
    base_probability: float  # Original model prediction
    is_exploration: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RLMetrics:
    cumulative_reward: float = 0.0
    total_impressions: int = 0
    total_likes: int = 0
    total_dislikes: int = 0
    exploration_count: int = 0
    exploitation_count: int = 0

    @property
    def average_reward(self) -> float:
        return self.cumulative_reward / max(1, self.total_impressions)

    @property
    def ctr(self) -> float:
        total_feedback = self.total_likes + self.total_dislikes
        return total_feedback / max(1, self.total_impressions)

    @property
    def like_rate(self) -> float:
        total_feedback = self.total_likes + self.total_dislikes
        return self.total_likes / max(1, total_feedback)

    @property
    def exploration_rate(self) -> float:
        return self.exploration_count / max(1, self.total_impressions)


class ThompsonSamplingAgent:
    """
    Contextual Multi-Armed Bandit agent using Thompson Sampling.

    This agent learns to recommend celebrities based on:
    - Historical feedback (likes/dislikes)
    - User context (facial embeddings)
    - Exploration-exploitation tradeoff

    Algorithm:
    1. For each celebrity, maintain Beta(α, β) distribution
    2. Sample probability θ ~ Beta(α, β) for each celebrity
    3. Adjust samples based on context similarity
    4. Select top-K celebrities with highest adjusted samples
    5. Update distributions based on feedback
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        context_weight: float = 0.3,
        exploration_bonus: float = 0.1,
        decay_rate: float = 1.0,
        min_impressions: int = 5,
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.context_weight = context_weight
        self.exploration_bonus = exploration_bonus
        self.decay_rate = decay_rate
        self.min_impressions = min_impressions

        self.celebrity_states: Dict[int, CelebrityState] = {}
        self.celebrity_name_to_id: Dict[str, int] = {}
        self.metrics = RLMetrics()

        self.action_history: List[RLAction] = []
        self.max_history = 1000

        logger.info(
            f"Initialized Thompson Sampling agent with α={alpha_prior}, "
            f"β={beta_prior}, context_weight={context_weight}"
        )

    def register_celebrity(self, celebrity_id: int, celebrity_name: str):
        if celebrity_id not in self.celebrity_states:
            self.celebrity_states[celebrity_id] = CelebrityState(
                celebrity_id=celebrity_id,
                celebrity_name=celebrity_name,
                alpha=self.alpha_prior,
                beta=self.beta_prior,
            )
            self.celebrity_name_to_id[celebrity_name] = celebrity_id
            logger.debug(f"Registered celebrity: {celebrity_name} (id={celebrity_id})")

    def _compute_context_affinity(
        self, celebrity_id: int, context: Optional[np.ndarray]
    ) -> float:
        if context is None or celebrity_id not in self.celebrity_states:
            return 1.0

        state = self.celebrity_states[celebrity_id]

        if not state.context_embeddings:
            return 1.0  # Neutral for new celebrities

        # Compute cosine similarity with all historical contexts
        similarities = []
        for hist_context in state.context_embeddings[-10:]:  # Last 10 contexts
            # Cosine similarity
            norm_context = context / (np.linalg.norm(context) + 1e-8)
            norm_hist = hist_context / (np.linalg.norm(hist_context) + 1e-8)
            sim = np.dot(norm_context, norm_hist)
            similarities.append(sim)

        # Average similarity
        avg_similarity = np.mean(similarities)

        # Map from [-1, 1] to [1 - weight, 1 + weight]
        affinity = 1.0 + avg_similarity * self.context_weight

        return max(0.5, min(1.5, affinity))

    def _compute_exploration_bonus(self, celebrity_id: int) -> float:
        if celebrity_id not in self.celebrity_states:
            return self.exploration_bonus

        state = self.celebrity_states[celebrity_id]

        if state.total_impressions < self.min_impressions:
            return self.exploration_bonus * (
                1.0 - state.total_impressions / self.min_impressions
            )

        return 0.0

    def select_celebrities(
        self,
        candidate_celebrities: List[Tuple[int, str, float]],
        context: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> List[Tuple[int, str, float, bool]]:
        # Register any new celebrities
        for celeb_id, celeb_name, _ in candidate_celebrities:
            self.register_celebrity(celeb_id, celeb_name)

        # Sample from Beta distributions and adjust
        samples = []
        for celeb_id, celeb_name, base_prob in candidate_celebrities:
            state = self.celebrity_states[celeb_id]

            # Thompson Sampling: sample from Beta distribution
            sampled_prob = state.sample_probability()

            # Adjust with context affinity
            context_affinity = self._compute_context_affinity(celeb_id, context)
            sampled_prob *= context_affinity

            # Add exploration bonus
            exploration_bonus = self._compute_exploration_bonus(celeb_id)
            sampled_prob += exploration_bonus

            # Determine if this is exploration or exploitation
            is_exploration = state.total_impressions < self.min_impressions

            samples.append(
                (celeb_id, celeb_name, sampled_prob, base_prob, is_exploration)
            )

        # Sort by adjusted probability and select top-K
        samples.sort(key=lambda x: x[2], reverse=True)
        top_samples = samples[:top_k]

        # Track actions
        for celeb_id, celeb_name, adj_prob, base_prob, is_exploration in top_samples:
            action = RLAction(
                celebrity_id=celeb_id,
                celebrity_name=celeb_name,
                sampled_probability=adj_prob,
                base_probability=base_prob,
                is_exploration=is_exploration,
            )
            self.action_history.append(action)

            # Keep history bounded
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)

            # Update impression count
            state = self.celebrity_states[celeb_id]
            state.total_impressions += 1
            self.metrics.total_impressions += 1

            if is_exploration:
                self.metrics.exploration_count += 1
            else:
                self.metrics.exploitation_count += 1

        # Store context for future affinity calculations
        if context is not None:
            for celeb_id, _, _, _, _ in top_samples:
                state = self.celebrity_states[celeb_id]
                state.context_embeddings.append(context.copy())
                if len(state.context_embeddings) > 20:
                    state.context_embeddings.pop(0)

        return [(cid, name, prob, is_exp) for cid, name, prob, _, is_exp in top_samples]

    def update(
        self,
        celebrity_id: int,
        reward: float,
        context: Optional[np.ndarray] = None,
    ):
        if celebrity_id not in self.celebrity_states:
            logger.warning(f"Received feedback for unknown celebrity {celebrity_id}")
            return

        state = self.celebrity_states[celebrity_id]

        if reward > 0:
            state.alpha += reward * self.decay_rate
            state.total_likes += 1
            self.metrics.total_likes += 1
        elif reward < 0:
            state.beta += abs(reward) * self.decay_rate
            state.total_dislikes += 1
            self.metrics.total_dislikes += 1

        state.last_updated = datetime.now()

        self.metrics.cumulative_reward += reward

        logger.info(
            f"Updated {state.celebrity_name}: α={state.alpha:.2f}, β={state.beta:.2f}, "
            f"mean_prob={state.mean_probability:.3f}, uncertainty={state.uncertainty:.4f}"
        )

    def get_celebrity_stats(self, celebrity_id: int) -> Optional[Dict]:
        if celebrity_id not in self.celebrity_states:
            return None

        state = self.celebrity_states[celebrity_id]
        return {
            "celebrity_id": state.celebrity_id,
            "celebrity_name": state.celebrity_name,
            "alpha": state.alpha,
            "beta": state.beta,
            "mean_probability": state.mean_probability,
            "uncertainty": state.uncertainty,
            "total_impressions": state.total_impressions,
            "total_likes": state.total_likes,
            "total_dislikes": state.total_dislikes,
            "last_updated": state.last_updated.isoformat()
            if state.last_updated
            else None,
        }

    def get_global_stats(self) -> Dict:
        return {
            "total_celebrities": len(self.celebrity_states),
            "cumulative_reward": self.metrics.cumulative_reward,
            "average_reward": self.metrics.average_reward,
            "total_impressions": self.metrics.total_impressions,
            "total_likes": self.metrics.total_likes,
            "total_dislikes": self.metrics.total_dislikes,
            "ctr": self.metrics.ctr,
            "like_rate": self.metrics.like_rate,
            "exploration_rate": self.metrics.exploration_rate,
            "exploration_count": self.metrics.exploration_count,
            "exploitation_count": self.metrics.exploitation_count,
        }

    def get_top_celebrities(self, top_k: int = 10) -> List[Dict]:
        celebrities = []
        for state in self.celebrity_states.values():
            celebrities.append(
                {
                    "celebrity_id": state.celebrity_id,
                    "celebrity_name": state.celebrity_name,
                    "mean_probability": state.mean_probability,
                    "total_impressions": state.total_impressions,
                    "total_likes": state.total_likes,
                }
            )

        celebrities.sort(key=lambda x: x["mean_probability"], reverse=True)
        return celebrities[:top_k]

    def reset(self):
        """Reset agent to initial state"""
        self.celebrity_states.clear()
        self.celebrity_name_to_id.clear()
        self.metrics = RLMetrics()
        self.action_history.clear()
        logger.info("Reset RL agent to initial state")

    def load_from_feedback(self, feedbacks: List[Dict]):
        logger.info(f"Loading RL agent from {len(feedbacks)} historical feedbacks")

        for feedback in feedbacks:
            celeb_id = feedback.get("celebrity_id")
            celeb_name = feedback.get("celebrity_name")
            feedback_type = feedback.get("feedback_type")

            if celeb_id and celeb_name:
                self.register_celebrity(celeb_id, celeb_name)

                reward = 1.0 if feedback_type == "like" else -1.0
                self.update(celeb_id, reward)

        logger.info(
            f"Loaded {len(self.celebrity_states)} celebrities, "
            f"cumulative_reward={self.metrics.cumulative_reward:.2f}"
        )


# Global RL agent instance
rl_agent = ThompsonSamplingAgent(
    alpha_prior=1.0,
    beta_prior=1.0,
    context_weight=0.3,
    exploration_bonus=0.1,
    decay_rate=1.0,
    min_impressions=5,
)
