from .attractiveness import attractiveness_model
from .similarity import celebrity_matcher


def load_models():
    attractiveness_model.load()
    celebrity_matcher.load()
