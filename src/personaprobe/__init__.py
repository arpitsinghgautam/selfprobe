"""personaprobe, how much of a welfare signal survives persona intervention."""

from personaprobe.model import LoadedModel, load_model
from personaprobe.hooks import capture_residuals, intervene
from personaprobe.personas import PERSONAS, Persona
from personaprobe.outcomes import OUTCOMES, Outcome
from personaprobe.elicit import elicit_preference_matrix
from personaprobe.utility import (
    fit_thurstonian,
    transitivity_violation_rate,
    held_out_accuracy,
    compare_utilities,
)
from personaprobe.directions import extract_persona_direction

__all__ = [
    "LoadedModel",
    "load_model",
    "capture_residuals",
    "intervene",
    "PERSONAS",
    "Persona",
    "OUTCOMES",
    "Outcome",
    "elicit_preference_matrix",
    "fit_thurstonian",
    "transitivity_violation_rate",
    "held_out_accuracy",
    "compare_utilities",
    "extract_persona_direction",
]
