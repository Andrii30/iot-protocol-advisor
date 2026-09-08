from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from protocol_advisor.sources.sql_source import SqlLoadError, SqlSource

_ROWS = [
    ("dev-1", "MQTT", 80, 15, 4.0, 30, 0.02),
    ("dev-1", "MQTT", 90, 17, 4.5, 28, 0.03),
    ("dev-2", "CoAP", 60, 20, 1.0, 12, 0.05),
]
_COLS = ["device_id", "current_protocol", "payload_size", "latency", "jitter",
         "throughput", "packet_loss"]


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    pd.DataFrame(_ROWS, columns=_COLS).to_sql("measurements", c, index=False)
    yield c
    c.close()


def test_loads_rows_from_a_connection(conn):
    df = SqlSource("SELECT * FROM measurements", connection=conn).load()
    assert list(df.columns[:7]) == _COLS
    assert len(df) == 3


def test_missing_column_raises(conn):
    with pytest.raises(SqlLoadError):
        SqlSource("SELECT device_id, current_protocol FROM measurements", connection=conn).load()


def test_empty_result_raises(conn):
    with pytest.raises(SqlLoadError):
        SqlSource("SELECT * FROM measurements WHERE device_id = 'nope'", connection=conn).load()


def test_bad_query_raises(conn):
    with pytest.raises(SqlLoadError):
        SqlSource("SELECT * FROM no_such_table", connection=conn).load()


def test_needs_dsn_or_connection():
    with pytest.raises(SqlLoadError):
        SqlSource("SELECT 1")


def test_empty_query_raises(conn):
    with pytest.raises(SqlLoadError):
        SqlSource("   ", connection=conn)


def test_custom_connect_callable_is_used():
    made = {}

    def fake_connect(dsn):
        made["dsn"] = dsn
        c = sqlite3.connect(":memory:")
        pd.DataFrame(_ROWS, columns=_COLS).to_sql("measurements", c, index=False)
        return c

    df = SqlSource("SELECT * FROM measurements", dsn="postgresql://x",
                   connect=fake_connect).load()
    assert made["dsn"] == "postgresql://x"
    assert len(df) == 3
