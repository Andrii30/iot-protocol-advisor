"""Load measurements from a PostgreSQL database (or any DB-API connection).

Install the driver: ``pip install "iot-protocol-advisor[db]"`` (pulls psycopg 3).

The query must return at least the columns in ``INPUT_COLUMNS``. For large
tables, pre-aggregate in SQL so only ~one row per device comes back, e.g.::

    SELECT device_id,
           mode() WITHIN GROUP (ORDER BY current_protocol) AS current_protocol,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY payload_size) AS payload_size,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY latency)      AS latency,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY jitter)       AS jitter,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY throughput)   AS throughput,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY packet_loss)  AS packet_loss
    FROM measurements
    WHERE ts > now() - interval '1 day'
    GROUP BY device_id
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from protocol_advisor.sources.base import INPUT_COLUMNS, DataSource


class SqlLoadError(RuntimeError):
    """Raised when the DB cannot be reached or the query result is unusable."""


def _default_connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SqlLoadError(
            'PostgreSQL support needs the driver. Install with:\n'
            '  pip install "iot-protocol-advisor[db]"'
        ) from exc
    return psycopg.connect(dsn)


class SqlSource(DataSource):
    def __init__(
        self,
        query: str,
        dsn: str | None = None,
        *,
        connection=None,
        connect: Callable[[str], object] = _default_connect,
    ):
        """Either pass a ready ``connection`` or a ``dsn`` to connect with.

        ``dsn`` is a libpq string / URL, e.g.
        ``postgresql://user:pass@host:5432/dbname``.
        """
        if not query or not query.strip():
            raise SqlLoadError("Query is empty.")
        if connection is None and not dsn:
            raise SqlLoadError("Provide either a connection or a dsn.")
        self.query = query
        self.dsn = dsn
        self._connection = connection
        self._connect = connect

    def load(self) -> pd.DataFrame:
        conn = self._connection
        opened_here = False
        try:
            if conn is None:
                try:
                    conn = self._connect(self.dsn)
                except SqlLoadError:
                    raise
                except Exception as exc:  # noqa: BLE001 - clean message for the UI
                    raise SqlLoadError(f"Cannot connect to the database: {exc}") from exc
                opened_here = True

            try:
                data = pd.read_sql_query(self.query, conn)
            except Exception as exc:  # noqa: BLE001
                raise SqlLoadError(f"Query failed: {exc}") from exc
        finally:
            if opened_here and conn is not None:
                conn.close()

        missing = [c for c in INPUT_COLUMNS if c not in data.columns]
        if missing:
            raise SqlLoadError(
                "Query result is missing required columns: " + ", ".join(missing)
            )
        data = data.dropna(subset=INPUT_COLUMNS)
        if data.empty:
            raise SqlLoadError("Query returned no rows with all required columns.")
        return data
