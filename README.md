# Facial Analysis Mobile App - PMLDL Project

## Project Overview

A mobile application that uses computer vision and reinforcement learning to analyze facial features for entertainment purposes. The app provides three main functionalities:
- **Facial Attractiveness Score**: Estimates an attractiveness rating based on facial features
- **Celebrity Look-alike Finder**: Identifies which famous celebrity the user resembles most with AI-powered matching
- **Reinforcement Learning System**: Advanced Thompson Sampling algorithm learns from user feedback to continuously improve predictions

## Technical Architecture

### Core ML Models
- **Attractiveness Classifier**: ResNet-50 CNN fine-tuned on SCUT-FBP5500 dataset
- **Celebrity Matcher**: Inception Resnet V1 pretrained on VGGFace2, fine-tuned on Open Famous People Faces dataset
  - **Reinforcement Learning Integration**: Thompson Sampling (Contextual Multi-Armed Bandit)
  - **Online Learning**: Learns from every user interaction in real-time
  - **Exploration-Exploitation**: Balances showing proven matches vs. discovering new ones
  - **Context-Aware**: Uses facial embeddings to personalize recommendations
  - **Bayesian Approach**: Maintains uncertainty estimates for each celebrity

### Infrastructure Stack
- **Backend**: FastAPI with PostgreSQL
- **Frontend**: Flutter mobile application
- **Deployment**: Docker containerization
- **Observability**: Prometheus metrics, OpenTelemetry tracing, MLflow experiment tracking

## Datasets

### SCUT-FBP5500 Dataset
- **Size**: 5,500 frontal faces (350×350px)
- **Diversity**: 2,000 Asian females, 2,000 Asian males, 750 Caucasian females, 750 Caucasian males
- **Labels**: Attractiveness scores (1.0-5.0) from human ratings
- **Split**: 60% training, 40% test sets

### Open Famous People Faces Dataset
- **Classes**: 258 celebrities with ≥5 images per class
- **Quality**: Varied image sizes and quality levels
- **Alignment**: Face-aligned using eye position and landmarks
- **Use Case**: Face re-identification and recognition

## Key Features

### Reinforcement Learning System (Thompson Sampling)

The Celebrity Matcher uses a state-of-the-art **Contextual Multi-Armed Bandit** approach with Thompson Sampling:

#### Algorithm Overview
- **Problem**: Each celebrity is a "bandit arm" to pull
- **Context**: User's facial features (512-dim embedding)
- **Action**: Recommend top-K celebrities
- **Reward**: +1 for like, -1 for dislike
- **Goal**: Maximize cumulative user satisfaction

#### How It Works
1. **Beta Distributions**: Each celebrity has Beta(α, β) representing success probability
   - α = number of likes + prior
   - β = number of dislikes + prior
2. **Thompson Sampling**: Sample from each Beta distribution
3. **Context Adjustment**: Adjust samples based on facial similarity
4. **Exploration Bonus**: Boost under-explored celebrities
5. **Select Top-K**: Show celebrities with highest adjusted samples
6. **Update Beliefs**: Increment α or β based on feedback

#### Advantages
- **Principled Exploration**: Naturally balances trying new vs. proven celebrities
- **Uncertainty Quantification**: Knows what it doesn't know
- **Contextual**: Personalizes based on facial features
- **Online Learning**: Updates instantly with each feedback
- **Bayesian Optimality**: Theoretically grounded in decision theory

### Simple Feedback System (Fallback)
For comparison, also includes a simpler heuristic feedback system:
- **Aggregate Statistics**: Tracks likes/dislikes per celebrity
- **Probability Reweighting**: Adjusts predictions by up to ±20%
- **No Exploration**: Purely exploitative

### API Endpoints

#### Feedback Endpoints
- `POST /ml/users/me/feedback` - Submit like/dislike (updates RL agent)
- `GET /ml/users/me/feedback` - Retrieve user's feedback history
- `GET /ml/users/me/feedback/stats` - Get user's feedback statistics
- `GET /ml/celebrities/{id}/feedback/stats` - Get celebrity's feedback statistics

#### RL Agent Endpoints
- `GET /ml/rl/stats` - Global RL agent statistics (cumulative reward, CTR, etc.)
- `GET /ml/rl/celebrity/{id}` - Celebrity's Beta distribution parameters (α, β)
- `GET /ml/rl/top-celebrities` - Top celebrities by RL agent's belief
- `GET /ml/rl/metrics` - Detailed performance metrics (regret, exploration rate)
- `POST /ml/rl/reset` - Reset RL agent (admin only)

#### Admin Endpoints
- `POST /ml/admin/reload-feedback-weights` - Reload feedback from database

## Repository Structure

```
├── src/
│   ├── application/
│   │   └── services/          # Business logic layer
│   │       ├── ml.py          # ML inference service
│   │       ├── feedback.py    # Feedback management service
│   │       └── ...
│   ├── infrastructure/
│   │   ├── ml_models/         # PyTorch ML models
│   │   │   ├── attractiveness/
│   │   │   └── similarity/    # Celebrity matcher with feedback
│   │   ├── database/          # SQLAlchemy models
│   │   └── repositories/      # Data access layer
│   └── interfaces/
│       ├── api/v1/            # FastAPI REST API
│       └── cli/               # Command-line interface
├── datasets/                  # Training data
├── models/                    # Trained .pt model files
├── alembic/                   # Database migrations
└── tests/                     # Test suite
```
## Observability & Monitoring

The application includes comprehensive observability features:

### Metrics (Prometheus)
- **HTTP Metrics**: Request latency, status codes, throughput
- **ML Inference**: Prediction time, model load time
- **Database**: Query performance
- **RL Agent Metrics**:
  - `rl_cumulative_reward`: Total likes - dislikes
  - `rl_average_reward`: Mean reward per impression
  - `rl_exploration_rate`: % of exploratory actions
  - `rl_ctr`: Click-through rate (feedback rate)
  - `rl_like_rate`: % of positive feedback
  - `rl_celebrity_alpha`: Beta distribution α per celebrity
  - `rl_celebrity_beta`: Beta distribution β per celebrity
  - `rl_celebrity_mean_probability`: Expected success rate

Access Prometheus metrics at `/metrics` endpoint.

### Distributed Tracing (OpenTelemetry)
- End-to-end request tracing across services
- ML model inference spans with timing
- Database query tracing
- Automatic context propagation

Configure OTLP exporter via environment variables:
```bash
OTLP_ENDPOINT=http://jaeger:4317
OTLP_SERVICE_NAME=beauty-classifier-api
```

### Experiment Tracking (MLflow)
- Model training runs with hyperparameters
- Metrics tracking (accuracy, F1-score, loss)
- Model versioning and artifact storage
- Automatic model registration

MLflow tracks:
- Training epochs and learning curves
- Model architecture and parameters
- Dataset statistics
- Best model checkpoints

## Development Setup

### Prerequisites
- [make](https://www.gnu.org/software/make/), [uv](https://www.uvproject.xyz/), [docker](https://docs.docker.com/get-docker/), [docker-compose](https://docs.docker.com/compose/install/)

### Quick Start
1. `make setup`
2. `make run`
Run `make help` for more commands

### Testing
Run `make test` to execute test suite

### Database Migrations
```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Project Timeline
5 two-week sprints concluding November 25, 2025, focusing on iterative development of ML models, mobile integration, and deployment infrastructure.