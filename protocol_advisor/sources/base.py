"""The data-source interface.

A source turns some external thing (a file, a DB query, an SNMP poll) into a
DataFrame of per-device measurements. To add one, implement ``load()`` and
return a frame with at least these columns::

    device_id, current_protocol, payload_size, latency, jitter, throughput, packet_loss
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

INPUT_COLUMNS: list[str] = [
    "device_id",
    "current_protocol",
    "payload_size",
    "latency",
    "jitter",
    "throughput",
    "packet_loss",
]


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame containing at least ``INPUT_COLUMNS``."""
        raise NotImplementedError
