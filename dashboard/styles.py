import re
import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .stApp {
        background: radial-gradient(circle at top left, #16232A 0%, #0D1417 60%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B1518;
        border-right: 1px solid #1C2A2E;
    }
    section[data-testid="stSidebar"] * { color: #C7D1D3 !important; }

    [data-testid="stSidebarNav"] ul { padding-top: 4px; }
    [data-testid="stSidebarNav"] a {
        border-radius: 10px;
        padding: 10px 14px !important;
        margin: 2px 10px;
        font-weight: 500;
        font-size: 14px;
        transition: background 0.15s ease, transform 0.1s ease;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: #16262B;
        transform: translateX(2px);
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, #1B4A52 0%, #163A40 100%);
        box-shadow: inset 3px 0 0 #A8F03B;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] span {
        color: #C9F98A !important;
        font-weight: 600;
    }

    /* Sticky page header (use with page_header() component) */
    .app-header {
        position: sticky; top: 0; z-index: 999;
        backdrop-filter: blur(14px);
        background: rgba(13, 20, 23, 0.75);
        border-bottom: 1px solid #1C2A2E;
        padding: 18px 4px 16px 4px;
        margin: -1rem -1rem 1.5rem -1rem;
        padding-left: 1rem; padding-right: 1rem;
    }
    .app-header-eyebrow {
        display: flex; align-items: center; gap: 6px;
        color: #7FE38C; font-size: 12px; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px;
    }
    .app-header-eyebrow .dot {
        width: 7px; height: 7px; border-radius: 50%; background: #4ADE80;
        box-shadow: 0 0 8px #4ADE80; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .app-header-title {
        font-size: 30px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;
        margin: 0;
    }
    .app-header-sub { color: #8B9AA0; font-size: 14px; margin-top: 4px; }

    /* Buttons */
    .stButton > button {
        background: #16262B;
        color: #E8EDEE;
        border: 1px solid #253338;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #1B4A52; border-color: #33818D; color: #FFFFFF;
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #A8F03B, #7FD93B);
        color: #0D160A; border: none; border-radius: 10px; font-weight: 700;
    }
    .stDownloadButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }

    /* Selectbox / inputs */
    [data-testid="stSelectbox"] > div > div, [data-baseweb="select"] > div {
        background-color: #16262B !important; border-color: #253338 !important;
        border-radius: 10px !important; color: #E8EDEE !important;
    }
    input, textarea {
        background-color: #16262B !important; color: #E8EDEE !important; border-radius: 10px !important;
    }

    [data-testid="stChatInput"] {
        background-color: #16262B; border: 1px solid #253338; border-radius: 16px;
    }
    [data-testid="stChatInput"] textarea { color: #E8EDEE !important; }

    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #253338; }

    div[data-testid="stAlert"] {
        background-color: #16262B; border-radius: 10px; border: 1px solid #253338; color: #C7D1D3;
    }

    /* KPI cards — glass effect */
    .kpi-card {
        background: linear-gradient(160deg, rgba(33,37,43,0.9), rgba(22,25,29,0.9));
        border-radius: 16px; padding: 20px 22px;
        border: 1px solid #253338;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #33818D; }
    .kpi-label { color: #8B9AA0; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 8px; }
    .kpi-value { color: #FFFFFF; font-size: 26px; font-weight: 800; line-height: 1.2; word-break: break-word; }
    .kpi-sub { color: #6B7A80; font-size: 12px; margin-top: 8px; }

    .badge-good { background: #16311D; color: #A8F03B; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; white-space: nowrap; }
    .badge-moderate { background: #332711; color: #F0C93B; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; white-space: nowrap; }
    .badge-unhealthy { background: #331414; color: #F0603B; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; white-space: nowrap; }

    /* Chat bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #17767F, #14606B); color: white;
        padding: 12px 18px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 75%; margin-left: auto; text-align: right;
        box-shadow: 0 4px 14px rgba(20,96,107,0.35);
    }
    .chat-bubble-ai {
        background: #16262B; color: #E8EDEE; padding: 12px 18px;
        border-radius: 18px 18px 18px 4px; margin: 8px 0; max-width: 75%;
        border: 1px solid #253338;
    }

    /* Report card */
    .report-card {
        background: linear-gradient(160deg, rgba(33,37,43,0.92), rgba(20,23,27,0.92));
        border-radius: 18px; padding: 36px;
        border: 1px solid #253338; color: #D6DCDE; line-height: 1.7;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .report-card h1, .report-card h2, .report-card h3 { color: #FFFFFF !important; margin-top: 24px; margin-bottom: 10px; }
    .report-card h2 { color: #7FE38C !important; font-size: 18px; }
    .report-card table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    .report-card th, .report-card td { border: 1px solid #253338; padding: 9px 14px; text-align: left; font-size: 14px; }
    .report-card th { background: #16262B; color: #A8F03B; font-weight: 700; }
    .report-card strong { color: #FFFFFF; }

    /* Suggested question chips */
    .stButton > button[kind="secondary"] { text-align: left; }

    h1, h2, h3 { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    """Sticky, glass-blur page header. Call right after inject_css()."""
    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-eyebrow"><span class="dot"></span>{eyebrow}</div>
        <div class="app-header-title">{title}</div>
        <div class="app-header-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def aqi_badge(aqi):
    if aqi is None:
        return '<span class="badge-moderate">N/A</span>'
    if aqi <= 50:
        return f'<span class="badge-good">Good · AQI {aqi}</span>'
    elif aqi <= 100:
        return f'<span class="badge-moderate">Moderate · AQI {aqi}</span>'
    else:
        return f'<span class="badge-unhealthy">Unhealthy · AQI {aqi}</span>'


def markdown_to_html(text: str) -> str:
    lines = text.split("\n")
    html_lines = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        html_lines.append("<table>")
        for i, row in enumerate(table_rows):
            tag = "th" if i == 0 else "td"
            cells = "".join(f"<{tag}>{c}</{tag}>" for c in row)
            html_lines.append(f"<tr>{cells}</tr>")
        html_lines.append("</table>")
        table_rows = []
        in_table = False

    for raw in lines:
        line = raw.strip()
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            table_rows.append(cells)
            in_table = True
            continue
        else:
            if in_table:
                flush_table()
        if not line or line == "---":
            html_lines.append("<br>")
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            level = len(m.group(1)) + 2
            html_lines.append(f"<h{level}>{m.group(2)}</h{level}>")
            continue
        bold_only = re.match(r"^\*\*(.+?)\*\*:?$", line)
        if bold_only:
            html_lines.append(f"<h4>{bold_only.group(1)}</h4>")
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        if re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line)
            html_lines.append(f"<li>{content}</li>")
            continue
        html_lines.append(f"<p>{line}</p>")

    flush_table()
    return "\n".join(html_lines)