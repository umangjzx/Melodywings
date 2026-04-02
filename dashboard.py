"""
dashboard.py — Advanced Streamlit Safety Dashboard
MelodyWings Guard | Real-Time Content Safety System

This dashboard provides:
  - complete joined alert details (chat/video/audio specific)
  - multi-dimensional filtering
  - trend and breakdown visualizations
  - drilldown view per alert
  - CSV/JSON export for filtered results

Run with: streamlit run dashboard.py
"""

import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from database import get_db

st.set_page_config(
    page_title="MelodyWings Guard - Safety Dashboard",
    page_icon="MG",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOXICITY_THRESHOLD = 0.75
STRONG_NEGATIVE_THRESHOLD = 0.98
NSFW_THRESHOLD = 0.70
FLAGGED_VIDEO_EMOTIONS = {"disgust", "angry"}


def apply_theme() -> None:
    """Apply a clean, high-contrast visual theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(42, 157, 143, 0.08) 0, transparent 35%),
                radial-gradient(circle at 100% 10%, rgba(231, 111, 81, 0.08) 0, transparent 35%),
                linear-gradient(180deg, #f7faf9 0%, #f1f4f7 100%);
            color: #11212d;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        p, div, label, span {
            font-family: 'Source Sans 3', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fdfefe 0%, #edf4f3 100%);
            border-right: 1px solid #d8e4e1;
        }

        .mw-header {
            background: linear-gradient(120deg, #1d3557 0%, #2a9d8f 45%, #f4a261 100%);
            border-radius: 16px;
            padding: 22px 28px;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(17, 33, 45, 0.18);
        }

        .mw-header h1 {
            margin: 0;
            color: #ffffff;
            font-size: 2rem;
        }

        .mw-header p {
            margin: 4px 0 0;
            color: rgba(255, 255, 255, 0.92);
            font-size: 1rem;
        }

        .section-label {
            margin-top: 14px;
            margin-bottom: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #2a5a55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(value: Any) -> float | None:
    """Convert value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_audio_reasons(message: str) -> list[str]:
    """Extract comma-separated audio reason tokens from message text."""
    if not message:
        return []
    return [part.strip() for part in message.split(",") if part.strip()]


def _derive_reason_tags(row: dict[str, Any]) -> list[str]:
    """Build standardized reason tags from joined detail fields."""
    source = row.get("source", "")
    reasons: list[str] = []

    if source in ("chat", "transcript"):
        if row.get("has_profanity"):
            reasons.append("profanity")

        pii_types = row.get("pii_types") or []
        if isinstance(pii_types, list):
            for pii_type in pii_types:
                if pii_type:
                    reasons.append(f"pii:{pii_type}")

        toxicity_score = _safe_float(row.get("toxicity_score"))
        if toxicity_score is not None and toxicity_score >= TOXICITY_THRESHOLD:
            reasons.append("toxicity:toxic")

        sentiment = (row.get("sentiment") or "").lower()
        sentiment_score = _safe_float(row.get("sentiment_score"))
        if (
            sentiment == "negative"
            and sentiment_score is not None
            and sentiment_score > STRONG_NEGATIVE_THRESHOLD
        ):
            reasons.append("strong_negative_sentiment")

    elif source == "video_frame":
        nsfw_label = (row.get("nsfw_label") or "").lower()
        nsfw_score = _safe_float(row.get("nsfw_score"))
        if nsfw_label == "nsfw" and nsfw_score is not None and nsfw_score >= NSFW_THRESHOLD:
            reasons.append("nsfw:nsfw")

        emotion = (row.get("emotion") or "").lower()
        if emotion in FLAGGED_VIDEO_EMOTIONS:
            reasons.append(f"emotion:{emotion}")

    elif source == "audio":
        reasons.extend(_parse_audio_reasons(row.get("message", "")))

    if row.get("flagged") and not reasons:
        reasons.append("flagged_unspecified")

    return sorted(dict.fromkeys(reasons))


def _content_text(row: dict[str, Any]) -> str:
    """Return the most useful text content field for a row."""
    for candidate in (
        row.get("chat_text"),
        row.get("message"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _primary_reason_key(reason_tags: list[str]) -> Optional[str]:
    """Map reason tag to confidence_by_reason key."""
    if not reason_tags:
        return None

    first = str(reason_tags[0]).lower()
    if first.startswith("toxicity:"):
        return "toxicity"
    if first.startswith("pii:"):
        return "pii"
    if first == "profanity":
        return "profanity"
    if "sentiment" in first:
        return "sentiment"
    if first.startswith("emotion:"):
        return "emotion"
    if first.startswith("nsfw:"):
        return "nsfw"
    return None


def _display_confidence(row: dict[str, Any]) -> float:
    """Display confidence aligned to the row's primary reason."""
    confidence_map = row.get("confidence_by_reason")
    if isinstance(confidence_map, dict):
        reason_key = _primary_reason_key(row.get("reason_tags") or [])
        if reason_key and reason_key in confidence_map:
            return float(_safe_float(confidence_map.get(reason_key)) or 0.0)
    return float(_safe_float(row.get("confidence")) or 0.0)


@st.cache_data(ttl=3)
def load_alerts_df() -> pd.DataFrame:
    """Load and normalize detailed alerts into a dataframe."""
    db = get_db()
    rows = db.get_alerts_detailed()
    if not rows:
        return pd.DataFrame()

    prepared: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["timestamp_dt"] = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        row["content_text"] = _content_text(row)
        row["reason_tags"] = _derive_reason_tags(row)
        row["reason_text"] = ", ".join(row["reason_tags"]) if row["reason_tags"] else "safe"
        row["confidence_value"] = _display_confidence(row)
        row["emotion"] = (row.get("emotion") or "none").lower()
        row["sentiment"] = (row.get("sentiment") or "").lower()
        row["flagged"] = bool(row.get("flagged"))
        preview = row["content_text"]
        if len(preview) > 140:
            preview = preview[:140] + "..."
        row["preview"] = preview
        prepared.append(row)

    df = pd.DataFrame(prepared)
    if "timestamp_dt" in df.columns:
        df = df.sort_values("timestamp_dt", ascending=False, na_position="last").reset_index(drop=True)
    return df


def render_header(total_rows: int, filtered_rows: int) -> None:
    """Render top dashboard header."""
    st.markdown(
        """
        <div class="mw-header">
            <h1>MelodyWings Guard</h1>
            <p>Operational safety console with full alert diagnostics and precision filtering</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Data snapshot: {filtered_rows} shown / {total_rows} total records | "
        f"Loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def render_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Render sidebar controls and return filter state."""
    st.sidebar.title("Filter Console")

    if df.empty:
        st.sidebar.info("No records found in database yet.")
        return {
            "source": [],
            "status": "All",
            "severity": [],
            "category": [],
            "reasons": [],
            "sentiments": [],
            "emotions": [],
            "confidence": (0.0, 1.0),
            "date_range": None,
            "search": "",
            "max_rows": 500,
            "auto_refresh": True,
            "refresh_sec": 5,
        }

    source_opts = sorted(df["source"].dropna().unique().tolist())
    severity_opts = sorted(df["severity"].dropna().unique().tolist())
    category_opts = sorted(df["category"].dropna().unique().tolist())
    sentiment_opts = sorted([s for s in df["sentiment"].dropna().unique().tolist() if s])
    emotion_opts = sorted([e for e in df["emotion"].dropna().unique().tolist() if e != "none"])

    all_reasons = sorted({tag for tags in df["reason_tags"] for tag in tags})

    st.sidebar.markdown("#### Core")
    selected_sources = st.sidebar.multiselect("Source", source_opts, default=source_opts)
    status = st.sidebar.radio("Flag status", ["All", "Flagged only", "Safe only"], index=0)
    selected_severity = st.sidebar.multiselect("Severity", severity_opts, default=severity_opts)
    selected_category = st.sidebar.multiselect("Category", category_opts, default=category_opts)

    st.sidebar.markdown("#### Semantics")
    selected_reasons = st.sidebar.multiselect("Reason tags", all_reasons, default=all_reasons)
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_opts, default=sentiment_opts)
    selected_emotions = st.sidebar.multiselect("Video emotion", emotion_opts, default=emotion_opts)

    st.sidebar.markdown("#### Numeric")
    confidence = st.sidebar.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)

    min_ts = df["timestamp_dt"].min()
    max_ts = df["timestamp_dt"].max()
    if pd.notna(min_ts) and pd.notna(max_ts):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_ts.date(), max_ts.date()),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
        )
    else:
        date_range = None

    search = st.sidebar.text_input("Search text", value="").strip().lower()
    max_rows = st.sidebar.slider("Max rows in table", 100, 5000, 1000, 100)

    st.sidebar.markdown("#### Refresh")
    auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
    refresh_sec = st.sidebar.slider("Refresh interval (seconds)", 2, 60, 5)
    if st.sidebar.button("Refresh now"):
        st.rerun()

    st.sidebar.caption(f"Current time: {datetime.now().strftime('%H:%M:%S')}")

    return {
        "source": selected_sources,
        "status": status,
        "severity": selected_severity,
        "category": selected_category,
        "reasons": selected_reasons,
        "sentiments": selected_sentiments,
        "emotions": selected_emotions,
        "confidence": confidence,
        "date_range": date_range,
        "search": search,
        "max_rows": max_rows,
        "auto_refresh": auto_refresh,
        "refresh_sec": refresh_sec,
    }


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply all selected filters to dataframe."""
    if df.empty:
        return df

    out = df.copy()

    if filters["source"]:
        out = out[out["source"].isin(filters["source"])]
    else:
        out = out.iloc[0:0]

    if filters["status"] == "Flagged only":
        out = out[out["flagged"]]
    elif filters["status"] == "Safe only":
        out = out[~out["flagged"]]

    if filters["severity"]:
        out = out[out["severity"].isin(filters["severity"])]
    else:
        out = out.iloc[0:0]

    if filters["category"]:
        out = out[out["category"].isin(filters["category"])]
    else:
        out = out.iloc[0:0]

    conf_low, conf_high = filters["confidence"]
    out = out[(out["confidence_value"] >= conf_low) & (out["confidence_value"] <= conf_high)]

    selected_reasons = filters["reasons"]
    all_reasons = sorted({tag for tags in df["reason_tags"] for tag in tags})
    if all_reasons and selected_reasons != all_reasons:
        selected_set = set(selected_reasons)
        out = out[out["reason_tags"].apply(lambda tags: bool(selected_set.intersection(tags)))]

    selected_sentiments = filters["sentiments"]
    all_sentiments = sorted([s for s in df["sentiment"].dropna().unique().tolist() if s])
    if all_sentiments and selected_sentiments != all_sentiments:
        out = out[out["sentiment"].isin(selected_sentiments)]

    selected_emotions = filters["emotions"]
    all_emotions = sorted([e for e in df["emotion"].dropna().unique().tolist() if e != "none"])
    if all_emotions and selected_emotions != all_emotions:
        out = out[out["emotion"].isin(selected_emotions)]

    date_range = filters["date_range"]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)
        out = out[(out["timestamp_dt"] >= start_ts) & (out["timestamp_dt"] < end_ts)]

    query = filters["search"]
    if query:
        combined = (
            out["content_text"].fillna("")
            + " "
            + out["reason_text"].fillna("")
            + " "
            + out["source"].fillna("")
            + " "
            + out["severity"].fillna("")
        ).str.lower()
        out = out[combined.str.contains(query, na=False)]

    return out.head(filters["max_rows"])


def render_kpis(df: pd.DataFrame) -> None:
    """Render high-level KPI row."""
    total = len(df)
    flagged = int(df["flagged"].sum()) if not df.empty else 0
    safe = total - flagged
    flag_rate = (flagged / total * 100.0) if total else 0.0
    high_critical = int(df["severity"].isin(["high", "critical"]).sum()) if not df.empty else 0
    mean_conf_flagged = (
        float(df[df["flagged"]]["confidence_value"].mean())
        if flagged
        else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scanned", f"{total}")
    c2.metric("Flagged", f"{flagged}")
    c3.metric("Safe", f"{safe}")
    c4.metric("Flag Rate", f"{flag_rate:.1f}%")
    c5.metric("High/Critical", f"{high_critical}", delta=f"Avg conf {mean_conf_flagged:.2f}")


def render_charts(df: pd.DataFrame) -> None:
    """Render analytics charts for filtered data."""
    st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No records match current filters.")
        return

    left, right = st.columns(2)

    with left:
        source_counts = df.groupby("source", as_index=False).size().rename(columns={"size": "count"})
        fig_source = px.bar(
            source_counts,
            x="source",
            y="count",
            color="source",
            title="Records by Source",
            color_discrete_sequence=["#1d3557", "#2a9d8f", "#f4a261", "#e76f51"],
        )
        fig_source.update_layout(margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig_source, use_container_width=True)

    with right:
        severity_counts = df.groupby("severity", as_index=False).size().rename(columns={"size": "count"})
        fig_sev = px.pie(
            severity_counts,
            names="severity",
            values="count",
            title="Severity Split",
            hole=0.45,
            color="severity",
            color_discrete_map={
                "low": "#2a9d8f",
                "medium": "#e9c46a",
                "high": "#f4a261",
                "critical": "#e76f51",
            },
        )
        fig_sev.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sev, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        reason_counter = Counter()
        for tags in df["reason_tags"]:
            if tags:
                reason_counter.update(tags)
            else:
                reason_counter.update(["safe"])

        reason_df = pd.DataFrame(
            [{"reason": key, "count": value} for key, value in reason_counter.items()]
        ).sort_values("count", ascending=True)

        fig_reason = px.bar(
            reason_df.tail(12),
            x="count",
            y="reason",
            orientation="h",
            title="Top Reason Tags",
            color="count",
            color_continuous_scale="Teal",
        )
        fig_reason.update_layout(margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_reason, use_container_width=True)

    with right2:
        trend = df.copy()
        trend = trend[pd.notna(trend["timestamp_dt"])]
        if trend.empty:
            st.info("No timestamp data available for trend plot.")
        else:
            trend["minute_bin"] = trend["timestamp_dt"].dt.floor("min")
            trend["status"] = trend["flagged"].map({True: "flagged", False: "safe"})
            trend_df = (
                trend.groupby(["minute_bin", "status"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            fig_trend = px.line(
                trend_df,
                x="minute_bin",
                y="count",
                color="status",
                title="Event Trend Over Time",
                markers=True,
                color_discrete_map={"flagged": "#e76f51", "safe": "#2a9d8f"},
            )
            fig_trend.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    """Render filtered table and export controls."""
    st.markdown('<div class="section-label">Filtered Alert Log</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No records available for current filters.")
        return

    display = df.copy()
    display["flagged_label"] = display["flagged"].map({True: "YES", False: "NO"})
    display["timestamp_display"] = display["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    table_columns = [
        "id",
        "timestamp_display",
        "source",
        "flagged_label",
        "severity",
        "category",
        "confidence_value",
        "reason_text",
        "preview",
    ]

    st.dataframe(
        display[table_columns],
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "timestamp_display": st.column_config.TextColumn("Timestamp", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "flagged_label": st.column_config.TextColumn("Flagged", width="small"),
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "confidence_value": st.column_config.NumberColumn("Confidence", format="%.4f", width="small"),
            "reason_text": st.column_config.TextColumn("Reasons", width="large"),
            "preview": st.column_config.TextColumn("Preview", width="large"),
        },
    )

    export_df = display[table_columns].rename(columns={"timestamp_display": "timestamp"})
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    json_data = export_df.to_json(orient="records", indent=2)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download filtered CSV",
            data=csv_data,
            file_name=f"melodywings_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Download filtered JSON",
            data=json_data,
            file_name=f"melodywings_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_detail_panel(df: pd.DataFrame) -> None:
    """Render full detail drilldown for a selected alert row."""
    st.markdown('<div class="section-label">Alert Drilldown</div>', unsafe_allow_html=True)

    if df.empty:
        return

    id_to_row = {int(row["id"]): row for _, row in df.iterrows()}
    alert_ids = list(id_to_row.keys())

    selected_id = st.selectbox(
        "Inspect alert ID",
        options=alert_ids,
        index=0,
        format_func=lambda alert_id: (
            f"#{alert_id} | {id_to_row[alert_id]['source']} | "
            f"{id_to_row[alert_id]['severity']} | "
            f"{'FLAGGED' if id_to_row[alert_id]['flagged'] else 'SAFE'}"
        ),
    )

    row = id_to_row[int(selected_id)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Source", str(row.get("source", "")))
    c2.metric("Severity", str(row.get("severity", "")))
    c3.metric("Confidence", f"{row.get('confidence_value', 0.0):.4f}")

    st.write(f"Timestamp: {row.get('timestamp', '')}")
    st.write(f"Reasons: {row.get('reason_text', 'safe')}")
    st.write(f"Text: {row.get('content_text', '') or '[no text available]'}")

    source = row.get("source", "")
    if source in ("chat", "transcript"):
        st.markdown("**Chat/Transcript details**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Profanity", "Yes" if row.get("has_profanity") else "No")
        d2.metric("PII", "Yes" if row.get("has_pii") else "No")
        d3.metric("Sentiment", row.get("sentiment") or "n/a")
        d4.metric("Sentiment score", f"{_safe_float(row.get('sentiment_score')) or 0.0:.4f}")
        st.write(f"PII types: {', '.join(row.get('pii_types') or []) or 'none'}")
        entity_values = []
        for item in (row.get("entities") or []):
            if isinstance(item, str):
                item = {"text": item, "label": "unknown"}
            if isinstance(item, dict):
                entity_values.append(str(item.get("text", "")))
        st.write(f"Entities: {', '.join([e for e in entity_values if e]) or 'none'}")

    elif source == "video_frame":
        st.markdown("**Video frame details**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Frame number", f"{int(row.get('frame_number') or 0)}")
        d2.metric("Frame second", f"{_safe_float(row.get('timestamp_sec')) or 0.0:.3f}")
        d3.metric("NSFW label", str(row.get("nsfw_label") or "unknown"))
        d4.metric("Emotion", str(row.get("emotion") or "none"))

    elif source == "audio":
        st.markdown("**Audio details**")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Max dB", f"{_safe_float(row.get('max_volume_db')) or 0.0:.2f}")
        d2.metric("Mean dB", f"{_safe_float(row.get('mean_volume_db')) or 0.0:.2f}")
        d3.metric("Silence periods", f"{int(row.get('silence_count') or 0)}")
        d4.metric("Speech rate", f"{_safe_float(row.get('speech_rate_wpm')) or 0.0:.1f}")
        d5.metric("Speakers", f"{int(row.get('speaker_count') or 0)}")

    with st.expander("Raw row payload"):
        st.json({k: (v if not isinstance(v, pd.Timestamp) else v.isoformat()) for k, v in row.items()})


def main() -> None:
    """Render full dashboard."""
    apply_theme()

    data = load_alerts_df()
    filters = render_filters(data)
    filtered = apply_filters(data, filters)

    render_header(len(data), len(filtered))
    render_kpis(filtered)
    render_charts(filtered)
    render_table(filtered)
    render_detail_panel(filtered)

    if filters["auto_refresh"]:
        st.caption(f"Auto-refresh enabled. Next refresh in {filters['refresh_sec']}s.")
        time.sleep(filters["refresh_sec"])
        st.rerun()


if __name__ == "__main__":
    main()
