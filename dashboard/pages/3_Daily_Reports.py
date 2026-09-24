import json
import re
from pathlib import Path
from styles import inject_css, markdown_to_html
inject_css()
import streamlit as st
from api_client import ensure_logged_in, get, post
from fpdf import FPDF

st.set_page_config(page_title="AirSentinel — Daily Reports", layout="wide")

if not ensure_logged_in():
    st.stop()

from styles import page_header
page_header("Automated · Daily", "Daily Reports", "AI-synthesized environmental briefings per zone")

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reports"
# --- Sensor selector ---
try:
    sensors_resp = get("/api/v1/sensors/")
    sensor_ids = sensors_resp.get("sensors", [])
except Exception as e:
    st.error(f"Failed to load sensors: {e}")
    st.stop()

selected = st.selectbox("Select sensor", sensor_ids)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔄 Generate New Report"):
        with st.spinner("Generating report via AI agent..."):
            try:
                response = post(f"/api/v1/reports/{selected}")
                if response.get("status") == "success":
                    st.success("Report generated!")
                else:
                    st.error(
                        "Report generation failed: "
                        f"{response.get('error', 'unknown error')}"
                    )
                # The result above already reports success or failure.
            except Exception as e:
                st.error(f"Report generation failed: {e}")

# --- Load and render the report ---
report_path = REPORTS_DIR / f"report_{selected}.json"

if not report_path.exists():
    st.info(f"No report found yet for {selected}. Click 'Generate New Report' above.")
    st.stop()

with open(report_path, "r", encoding="utf-8") as f:
    report_data = json.load(f)

st.subheader(f"Report: {selected}")

markdown_content = report_data.get("report_body") or ""

if report_data.get("status") != "success" or not markdown_content:
    st.error(
        f"Report generation failed: {report_data.get('error', 'no content returned')}"
    )
    st.stop()

st.markdown(f'<div class="report-card">{markdown_to_html(markdown_content)}</div>', unsafe_allow_html=True)

# --- PDF export ---
# --- PDF export ---
UNICODE_REPLACEMENTS = {
    "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u202f": " ", "\u00a0": " ",
    "\u2192": "->", "\u2011": "-",
    "\u00b5": "u", "\u03bc": "u",      # micro sign AND greek mu — both used interchangeably by LLMs
    "\u00b2": "2", "\u00b3": "3",
    "\u2026": "...", "\u2022": "-",
    "\u2264": "<=", "\u2265": ">=",
    "\u00d7": "x", "\u00f7": "/",
    "\u00b0": " deg",
}


def clean_unicode(text: str) -> str:
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Drop any remaining unsupported characters silently instead of leaving "?" artifacts
    return text.encode("latin-1", "ignore").decode("latin-1")


def parse_markdown_lines(text: str):
    """Yields (kind, content) tuples: heading, bullet, table_row, hr, blank, paragraph."""
    lines = text.split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "---":
            yield ("blank", "")
            continue

        m = re.match(r"^#{1,3}\s+", line)
        if m:
            level = len(re.match(r"^(#{1,3})", line).group(1))
            content = re.sub(r"^#{1,3}\s+", "", line).replace("**", "")
            yield (f"heading{level}", content)
            continue

        # A line that is ENTIRELY bold (e.g. "**Executive Summary**") acts as a section heading
        # even without a # marker — this is how the LLM formats its section titles.
        bold_only = re.match(r"^\*\*(.+?)\*\*:?$", line)
        if bold_only:
            yield ("heading2", bold_only.group(1))
            continue

        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue  # skip separator rows

        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
            yield ("table_row", cells)
            continue

        if re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line).replace("**", "").replace("*", "")
            yield ("bullet", content)
            continue

        if re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^\d+\.\s+", "", line).replace("**", "").replace("*", "")
            yield ("bullet", content)
            continue

        if re.match(r"^\s{2,}[-*]\s+", raw_line):
            content = re.sub(r"^[-*]\s+", "", line).replace("**", "").replace("*", "")
            yield ("sub_bullet", content)
            continue

        content = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # inline bold -> plain (font weight handled separately)
        yield ("paragraph", content)


def markdown_to_pdf(markdown_content: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    text = clean_unicode(markdown_content)
    max_width = pdf.w - pdf.l_margin - pdf.r_margin

    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return
        n_cols = max(len(row) for row in table_buffer)
        col_width = max_width / n_cols
        for i, row in enumerate(table_buffer):
            row = row + [""] * (n_cols - len(row))
            pdf.set_font("Helvetica", "B" if i == 0 else "", 9)
            y_before = pdf.get_y()
            x_start = pdf.l_margin
            max_h = 6
            for cell in row:
                pdf.set_xy(x_start, y_before)
                pdf.multi_cell(col_width, 5, cell, border=1)
                max_h = max(max_h, pdf.get_y() - y_before)
                x_start += col_width
            pdf.set_xy(pdf.l_margin, y_before + max_h)
        table_buffer = []
        pdf.ln(4)

    for kind, content in parse_markdown_lines(text):
        if kind != "table_row":
            flush_table()

        if kind == "blank":
            pdf.ln(3)
        elif kind == "heading1":
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(20, 96, 107)
            pdf.multi_cell(max_width, 9, content)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        elif kind == "heading2":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 96, 107)
            pdf.multi_cell(max_width, 8, content)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif kind == "heading3":
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(max_width, 6, content)
            pdf.ln(1)
        elif kind == "bullet":
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(max_width - 4, 6, f"- {content}")
        elif kind == "sub_bullet":
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(pdf.l_margin + 10)
            pdf.multi_cell(max_width - 10, 6, f"- {content}")
        elif kind == "table_row":
            table_buffer.append(content)
        else:  # paragraph
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(max_width, 6, content)
            pdf.ln(1)

    flush_table()
    return bytes(pdf.output())


pdf_bytes = markdown_to_pdf(markdown_content)

st.download_button(
    label="⬇️ Download as PDF",
    data=pdf_bytes,
    file_name=f"report_{selected}.pdf",
    mime="application/pdf",
)