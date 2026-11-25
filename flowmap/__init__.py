"""flowmap-lab: a compact, runnable reference implementation of FLM / FMLM."""

__version__ = "0.1.0"

from .schedule import LogLinearSchedule, ScheduleLUT, make_schedule  # noqa: F401
from .model import make_model  # noqa: F401
from .flm import flm_loss, flm_sample, integrate  # noqa: F401
from .fmlm import flowmap_loss, one_step_sample  # noqa: F401
