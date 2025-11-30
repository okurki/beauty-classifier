# Testing Guide

## Overview

This document describes the testing strategy for the Celebrity Matcher feedback and RL system.

## Test Structure

```
tests/
├── unit/                          # Unit tests
│   └── test_rl_agent.py          # RL agent logic tests
├── integration/                   # Integration tests
│   └── api/
│       └── ml/
│           └── test_feedback_api.py  # Feedback API tests
└── TESTING.md                     # This file
```

## Running Tests

### Run All Tests
```bash
make test
```

### Run Specific Test File
```bash
pytest tests/unit/test_rl_agent.py -v
pytest tests/integration/api/ml/test_feedback_api.py -v
```

### Run With Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test
```bash
pytest tests/unit/test_rl_agent.py::test_agent_update_like -v
```

## Test Categories

### 1. Unit Tests (`tests/unit/test_rl_agent.py`)

Tests the core RL agent logic in isolation:

- **State Management**
  - `test_celebrity_state_initialization`
  - `test_celebrity_state_mean_probability`
  - `test_celebrity_state_uncertainty`
  - `test_celebrity_state_sampling`

- **Metrics**
  - `test_rl_metrics_initialization`
  - `test_rl_metrics_average_reward`
  - `test_rl_metrics_ctr`
  - `test_rl_metrics_like_rate`

- **Agent Operations**
  - `test_agent_initialization`
  - `test_agent_register_celebrity`
  - `test_agent_update_like`
  - `test_agent_update_dislike`
  - `test_agent_reset`

- **Thompson Sampling**
  - `test_agent_select_celebrities_basic`
  - `test_agent_select_celebrities_with_context`
  - `test_thompson_sampling_exploration`
  - `test_thompson_sampling_exploitation`

- **Context & Exploration**
  - `test_agent_exploration_bonus`
  - `test_agent_context_affinity`

- **Statistics**
  - `test_agent_get_celebrity_stats`
  - `test_agent_get_global_stats`
  - `test_agent_get_top_celebrities`

- **Data Loading**
  - `test_agent_load_from_feedback`
  - `test_agent_decay_rate`

### 2. Integration Tests (`tests/integration/api/ml/test_feedback_api.py`)

Tests the complete feedback flow through the API:

- **Feedback Creation**
  - `test_create_feedback_success` - Submit like feedback
  - `test_create_feedback_dislike` - Submit dislike feedback
  - `test_create_feedback_invalid_type` - Invalid feedback type
  - `test_create_feedback_unauthorized` - No authentication
  - `test_feedback_case_insensitive` - Case handling

- **Feedback Retrieval**
  - `test_get_user_feedbacks` - Get user's feedback history
  - `test_get_user_feedback_stats` - User statistics
  - `test_get_celebrity_feedback_stats` - Celebrity statistics

- **RL Agent Integration**
  - `test_rl_agent_stats` - Global RL statistics
  - `test_rl_agent_top_celebrities` - Top celebrities
  - `test_rl_metrics` - Detailed metrics
  - `test_feedback_updates_rl_agent` - Verify RL updates

- **Edge Cases**
  - `test_multiple_feedbacks_same_celebrity` - Accumulation
  - `test_celebrity_id_in_inference` - ID presence (TODO)

## Test Data

### Test Users
Tests create temporary users with pattern: `{test_name}_user`

### Test Celebrities
Tests use celebrity IDs 1-10 for consistency

### Test Inferences
Tests use inference IDs 1-100

## Assertions

### Feedback Response
```python
assert response.status_code == 200
assert data["user_id"] == user["id"]
assert data["feedback_type"] in ["like", "dislike"]
assert "id" in data
assert "timestamp" in data
```

### RL Stats
```python
assert "cumulative_reward" in data
assert "exploration_rate" in data
assert "total_impressions" in data
```

### Celebrity Stats
```python
assert "total" in data
assert "likes" in data
assert "dislikes" in data
assert "score" in data
```

## Mocking

### Database
Integration tests use real database (test database)

### ML Models
Unit tests mock model predictions

### RL Agent
Integration tests use real RL agent instance

## Coverage Goals

- **Unit Tests**: 90%+ coverage of RL agent code
- **Integration Tests**: All API endpoints tested
- **Edge Cases**: Error handling, validation, auth

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Commits to main
- Pre-deployment

### GitHub Actions Example
```yaml
- name: Run tests
  run: |
    make test
    pytest --cov=src --cov-report=xml
```

## Debugging Tests

### Verbose Output
```bash
pytest -v -s
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed
```bash
pytest --lf
```

### Print Statements
```bash
pytest -s  # Shows print() output
```

## Performance Tests

### Load Testing
```bash
# TODO: Add locust or similar
```

### Benchmark RL Agent
```python
def test_rl_agent_performance():
    agent = ThompsonSamplingAgent()
    # Measure selection time for 1000 celebrities
    start = time.time()
    agent.select_celebrities(candidates, top_k=10)
    elapsed = time.time() - start
    assert elapsed < 0.1  # Should be fast
```

## Known Issues

1. **Image Upload Tests**: Require actual image files
2. **Model Loading**: Tests may be slow on first run
3. **Database**: Requires PostgreSQL running

## Future Improvements

- [ ] Add property-based tests (Hypothesis)
- [ ] Add mutation testing
- [ ] Add visual regression tests
- [ ] Add load/stress tests
- [ ] Add E2E tests with Selenium
- [ ] Mock external dependencies better
- [ ] Add test fixtures for common scenarios

## Test Maintenance

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use descriptive test names
4. Add docstrings
5. Update this README

### Updating Tests

When changing functionality:
1. Update affected tests
2. Verify all tests pass
3. Check coverage didn't decrease
4. Update documentation

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Thompson Sampling Paper](https://arxiv.org/abs/1707.02038)

