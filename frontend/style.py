"""Custom CSS for Finance AI Analyst — Bloomberg-terminal inspired, zero AI slop."""

CSS = """
<style>
/* ============================================================
   Design tokens — OKLCH, terminal-inspired
   Mood: Bloomberg terminal at dawn, amber phosphor glow
   ============================================================ */

:root {
  --bg:              oklch(0.09 0.008 84);
  --surface:         oklch(0.13 0.006 84);
  --surface-raised:  oklch(0.14 0 0);
  --border:          oklch(0.20 0 0);
  --border-hover:    oklch(0.30 0 0);

  --ink:             oklch(0.93 0 0);
  --ink-secondary:   oklch(0.62 0 0);
  --ink-muted:       oklch(0.42 0 0);

  --amber:           oklch(0.72 0.16 82);
  --amber-dim:       oklch(0.55 0.12 82);
  --amber-bg:        oklch(0.18 0.03 82);

  --positive:        oklch(0.62 0.15 150);
  --positive-bg:     oklch(0.16 0.03 150);
  --negative:        oklch(0.58 0.20 22);
  --negative-bg:     oklch(0.16 0.03 22);

  --font-mono:       'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  --font-sans:       system-ui, -apple-system, 'Segoe UI', sans-serif;

  --radius:          4px;
  --radius-sm:       2px;
}

/* ---- global ---- */
.stApp {
  background: var(--bg);
}

/* ---- sidebar — tighter rhythm ---- */
[data-testid="stSidebar"] {
  background: var(--bg);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1 {
  font-size: 1.2rem !important;
  margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stMarkdown p {
  margin-bottom: 12px !important;
}
[data-testid="stSidebar"] .stSelectbox {
  margin-bottom: 8px;
}
[data-testid="stSidebar"] .stTextInput {
  margin-bottom: 8px;
}
[data-testid="stSidebar"] .stDivider {
  margin: 12px 0 !important;
}
[data-testid="stSidebar"] .stButton {
  margin-bottom: 6px;
}
[data-testid="stSidebar"] .stButton > button {
  border-radius: var(--radius);
  font-weight: 500;
  font-family: var(--font-sans);
  transition: background 150ms ease-out;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--ink);
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--surface-raised);
  border-color: var(--amber);
  color: var(--amber);
}

/* ---- primary button ---- */
.stButton > button[kind="primary"] {
  background: var(--amber);
  border: 1px solid var(--amber);
  color: var(--bg);
  font-weight: 600;
  transition: background 150ms ease-out;
}
.stButton > button[kind="primary"]:hover {
  background: oklch(0.78 0.16 82);
  border-color: oklch(0.78 0.16 82);
  color: var(--bg);
}

/* ---- secondary / normal buttons ---- */
.stButton > button {
  border-radius: var(--radius);
  font-family: var(--font-sans);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--ink);
  transition: background 150ms ease-out, border-color 150ms ease-out;
}
.stButton > button:hover {
  background: var(--surface-raised);
  border-color: var(--border-hover);
}

/* ---- text inputs ---- */
[data-testid="stTextInput"] input,
.stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.03em;
}
[data-testid="stTextInput"] input:focus,
.stTextInput input:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 2px var(--amber-bg) !important;
}

/* ---- select boxes ---- */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--ink) !important;
}

/* ---- metrics — flat surface, no visible card border ---- */
[data-testid="stMetric"] {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 12px 0 !important;
}
[data-testid="stMetric"] label {
  color: var(--ink-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
  font-family: var(--font-sans);
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-family: var(--font-mono);
  font-size: 0.85rem;
}

/* ---- tabs — clean underline style ---- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: transparent;
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 0;
  padding: 10px 20px;
  color: var(--ink-secondary);
  font-weight: 500;
  font-family: var(--font-sans);
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color 150ms ease-out;
}
.stTabs [aria-selected="true"] {
  color: var(--amber) !important;
  background: transparent !important;
  border-bottom: 2px solid var(--amber) !important;
}

/* ---- info / success / error boxes — no side stripes ---- */
[data-testid="stAlert"] {
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.stAlert[data-baseweb="notification"] {
  background: var(--surface);
  border: 1px solid var(--border);
}
div[data-testid="stAlert"][kind="info"] {
  border: 1px solid var(--border);
  background: var(--surface);
}
div[data-testid="stAlert"][kind="success"] {
  border: 1px solid var(--positive);
  background: var(--positive-bg);
}
div[data-testid="stAlert"][kind="error"] {
  border: 1px solid var(--negative);
  background: var(--negative-bg);
}

/* ---- progress bar ---- */
.stProgress > div > div {
  background: var(--amber);
  border-radius: var(--radius-sm);
}
.stProgress > div {
  background: var(--surface);
  border-radius: var(--radius-sm);
}

/* ---- title h1 — single solid color, no gradient text ---- */
h1 {
  font-weight: 700 !important;
  font-family: var(--font-sans) !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em;
  background: none !important;
  -webkit-text-fill-color: unset !important;
}
h2 {
  font-weight: 600 !important;
  font-family: var(--font-sans) !important;
  color: var(--ink) !important;
}
h3 {
  font-weight: 600 !important;
  font-family: var(--font-sans) !important;
  color: var(--ink) !important;
}

/* ---- landing page — vertical editorial index, not 3-card grid ---- */
.landing-hero {
  max-width: 680px;
  margin: 56px 0 48px 0;
}
.landing-hero h1 {
  font-size: 2.6rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 10px;
  line-height: 1.1;
}
.landing-hero .subtitle {
  color: var(--ink-secondary);
  font-size: 1rem;
  line-height: 1.6;
  max-width: 540px;
}

/* Editorial index list — vertical, asymmetric, large amber numerals */
.feature-list {
  margin: 36px 0 0 0;
  max-width: 600px;
}
.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 24px;
  padding: 18px 0;
  border-bottom: 1px solid var(--border);
  transition: border-color 200ms ease;
}
.feature-item:first-child {
  border-top: 1px solid var(--border);
}
.feature-item:hover {
  border-color: var(--border-hover);
}
.feature-num {
  font-family: var(--font-mono);
  font-size: 1.6rem;
  font-weight: 600;
  color: var(--amber);
  line-height: 1;
  min-width: 48px;
  font-variant-numeric: tabular-nums;
}
.feature-body h3 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 4px 0;
  color: var(--ink);
}
.feature-body p {
  font-size: 0.85rem;
  color: var(--ink-secondary);
  line-height: 1.55;
  margin: 0;
}

.footer-hint {
  color: var(--ink-muted);
  font-size: 0.8rem;
  margin-top: 36px;
  font-family: var(--font-mono);
  letter-spacing: 0.02em;
}

/* ---- debate comparison table ---- */
.debate-bull {
  padding: 6px 10px;
  font-size: 0.84rem;
  color: var(--ink-secondary);
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  line-height: 1.45;
}
.debate-bear {
  padding: 6px 10px;
  font-size: 0.84rem;
  color: var(--ink-secondary);
  border-bottom: 1px solid var(--border);
  border-left: 1px solid var(--border);
  vertical-align: top;
  line-height: 1.45;
}

/* ---- report header ---- */
.report-header {
  margin-bottom: 16px;
}
.report-header .report-symbol {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--amber-dim);
  letter-spacing: 0.04em;
}
.report-header .report-time {
  color: var(--ink-muted);
  font-size: 0.8rem;
  font-family: var(--font-mono);
}

/* ---- risk items — minimal row, no card wrapping ---- */
.risk-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  margin: 2px 0;
  border-bottom: 1px solid var(--border);
  font-family: var(--font-sans);
}
.risk-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: var(--radius-sm);
  font-size: 0.72rem;
  font-weight: 600;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
}
.risk-badge.low    { background: var(--positive-bg); color: var(--positive); }
.risk-badge.medium { background: var(--amber-bg);    color: var(--amber-dim); }
.risk-badge.high   { background: var(--negative-bg); color: var(--negative); }

/* ---- sentiment display ---- */
.sentiment-score-display {
  font-family: var(--font-mono);
  font-size: 3.2rem;
  font-weight: 700;
  color: var(--amber);
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.sentiment-label-display {
  font-size: 0.8rem;
  color: var(--ink-secondary);
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
}
.driver-tag {
  display: inline-block;
  padding: 3px 10px;
  margin: 2px 6px 2px 0;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 2px;
  font-size: 0.8rem;
  color: var(--ink-secondary);
  font-family: var(--font-sans);
  letter-spacing: 0.01em;
}

/* ---- data emphasis — amber highlights for key values ---- */
.data-highlight {
  color: var(--amber);
  font-family: var(--font-mono);
  font-weight: 600;
}

/* ---- dividers ---- */
hr, .stDivider {
  border-color: var(--border) !important;
}

/* ---- checkboxes ---- */
[data-testid="stCheckbox"] label {
  color: var(--ink-secondary) !important;
  font-family: var(--font-sans);
}

/* ---- selectbox labels ---- */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {
  color: var(--ink-secondary) !important;
  font-family: var(--font-sans);
  font-weight: 500 !important;
  font-size: 0.82rem !important;
}

/* ---- sidebar text ---- */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stCaption {
  color: var(--ink-secondary) !important;
}

/* ---- spinner ---- */
.stSpinner > div {
  border-color: var(--amber) !important;
}

/* ---- warnings in sidebar ---- */
[data-testid="stSidebar"] [data-testid="stAlert"] {
  background: var(--amber-bg);
  border: 1px solid var(--amber-dim);
}

/* ---- caption ---- */
.stCaption {
  color: var(--ink-muted) !important;
  font-family: var(--font-sans);
}

/* ---- links ---- */
a {
  color: var(--amber) !important;
}
a:hover {
  color: var(--amber) !important;
}

/* ---- responsive ---- */
@media (max-width: 768px) {
  .landing-hero {
    margin: 28px 0 28px 0;
  }
  .landing-hero h1 {
    font-size: 1.7rem;
  }
  .feature-item {
    gap: 16px;
  }
  .feature-num {
    font-size: 1.3rem;
    min-width: 36px;
  }
  .sentiment-score-display {
    font-size: 2.4rem;
  }
}
</style>
"""


def inject():
    """Inject custom CSS into the Streamlit app."""
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
