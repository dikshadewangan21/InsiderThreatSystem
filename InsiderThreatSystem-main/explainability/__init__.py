"""
explainability module

Production-ready Explainability Engine for Insider Threat Detection.
"""

from explainability.config import ExplainabilityConfig
from explainability.explanation_models import Explanation
from explainability.feature_importance import get_saliency_importance
from explainability.explainability_engine import ExplainabilityEngine
