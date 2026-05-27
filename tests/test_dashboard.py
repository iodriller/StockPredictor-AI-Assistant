from __future__ import annotations

import pandas as pd

from stockpredictor.ui.dashboard import _price_chart


def test_price_chart_uses_configured_ma_windows_and_result_levels() -> None:
    frame = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14],
            "High": [11, 12, 13, 14, 15],
            "Low": [9, 10, 11, 12, 13],
            "Close": [10, 11, 12, 13, 14],
            "Volume": [100, 110, 120, 130, 140],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    fig = _price_chart(frame, {"prior_high": 99.0, "session_open": 88.0}, ma_windows=[3])
    trace_names = [trace.name for trace in fig.data]
    annotation_texts = [annotation.text for annotation in fig.layout.annotations]

    assert "SMA_3" in trace_names
    assert "SMA_9" not in trace_names
    assert "prior_high: $99.00" in annotation_texts
    assert "session_open: $88.00" in annotation_texts
