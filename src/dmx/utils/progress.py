"""Report fitting and embedding progress as text, log records, and events.

Long-running helpers such as ``optimize``, ``best_of``, ``get_dpm_mixture``,
``htsne``, and ``humap`` accept ``verbose`` and ``progress_callback``
arguments and route their progress through :class:`ProgressReporter`:

- With ``verbose=True`` (the default) the existing progress text is written to
  the function's ``out`` stream (standard output unless overridden), exactly
  as before.
- With ``verbose=False`` nothing is written to standard output or standard
  error.
- Every progress message is also logged at ``DEBUG`` level on the calling
  module's logger (for example ``dmx.utils.estimation``). These records are
  silent unless logging is configured, e.g. with
  ``logging.basicConfig(level=logging.DEBUG)``.
- ``progress_callback``, when given, receives a :class:`ProgressEvent` for
  every iteration and trial, independent of ``verbose`` and ``print_iter``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import IO, Any, Callable, Mapping, Optional

import numpy as np

__all__ = ["ProgressEvent", "ProgressCallback", "ProgressReporter"]


@dataclass(frozen=True)
class ProgressEvent:
    """One structured progress update from a fitting or embedding helper.

    Attributes:
        source: Name of the reporting function, for example ``"optimize"``.
        event: ``"start"`` with the seed used, ``"iteration"`` for one update
            step, ``"trial"`` for the end of a restart in ``best_of``, or
            ``"result"`` for a final summary.
        iteration: One-based iteration number, when applicable.
        log_likelihood: Training log-likelihood after the step, when
            computed.
        delta: Change in training log-likelihood from the previous step.
        trial: One-based restart number for ``best_of`` style helpers.
        metrics: Source-specific values, such as
            ``validation_log_likelihood``, ``alpha``, ``kl_divergence``, or
            ``num_components``.
    """

    source: str
    event: str
    iteration: Optional[int] = None
    log_likelihood: Optional[float] = None
    delta: Optional[float] = None
    trial: Optional[int] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a JSON-serializable dictionary.

        NumPy scalars become Python numbers and NumPy arrays become lists.
        """
        rv = asdict(self)
        rv["metrics"] = dict(self.metrics)
        return {key: _to_builtin(value) for key, value in rv.items()}


def _to_builtin(value: Any) -> Any:
    """Convert NumPy values, recursively, to JSON-serializable builtins."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Route one call's progress to a text stream, a logger, and a callback.

    Args:
        source: Name of the reporting function, copied into every event.
        logger: Logger receiving each progress message at ``DEBUG`` level.
        out: Text stream for progress messages when ``verbose`` is true. When
            ``None``, the current ``sys.stdout`` is used.
        verbose: Whether progress messages are written to ``out``.
        callback: Optional function receiving every :class:`ProgressEvent`.
    """

    def __init__(
        self,
        source: str,
        logger: logging.Logger,
        out: Optional[IO[str]] = None,
        verbose: bool = True,
        callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Store the reporting destinations."""
        self.source = source
        self.logger = logger
        self.out = out
        self.verbose = verbose
        self.callback = callback

    def message(self, text: str) -> None:
        """Write ``text`` to the stream when verbose, and log it at DEBUG.

        Args:
            text: Message, including any trailing newline, written unchanged.
        """
        self.logger.debug("%s", text.rstrip("\n"))
        if self.verbose:
            stream = sys.stdout if self.out is None else self.out
            stream.write(text)

    def start(self, seed: Optional[int]) -> None:
        """Report the seed a call uses, without writing to ``out``.

        The seed is logged at ``DEBUG`` and sent to the callback as a
        ``"start"`` event with ``metrics={"seed": seed}``. ``seed`` is ``None``
        when the caller supplied its own random state.

        Args:
            seed: Seed the call's random state was built from, or ``None``.
        """
        self.logger.debug("%s seed=%s", self.source, seed)
        self.event("start", metrics={"seed": seed})

    def event(self, event: str, **kwargs: Any) -> None:
        """Send a :class:`ProgressEvent` to the callback, if one is set.

        Args:
            event: Event kind, such as ``"iteration"``.
            **kwargs: Remaining :class:`ProgressEvent` fields.
        """
        if self.callback is not None:
            self.callback(ProgressEvent(self.source, event, **kwargs))
