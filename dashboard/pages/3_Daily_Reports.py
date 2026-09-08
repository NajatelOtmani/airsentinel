import json
import re
from pathlib import Path

import streamlit as st
from api_client import ensure_logged_in, get, post
from fpdf import FPDF

st.set_page_config(page_title="AirSentinel — Daily Reports", layout="wide")

if not ensure_logged_in():
    st.stop()

st.title("📄 Daily Reports")

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
                post(f"/api/v1/reports/{selected}")
                st.success("Report generated!")
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

st.markdown(markdown_content)

# --- PDF export ---
UNICODE_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u202f": " ",
    "\u00a0": " ",
    "\u2192": "->",
    "\u2011": "-",
    "\u00b5": "u",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2026": "...",
}


def clean_unicode(text: str) -> str:
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def parse_markdown_lines(text: str):
    """Yields (kind, content) tuples: heading, bullet, table_row, hr, blank, paragraph."""
    lines = text.split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "---":
            yield ("blank", "")
        elif re.match(r"^#{1,3}\s+", line):
            level = len(re.match(r"^(#{1,3})", line).group(1))
            content = re.sub(r"^#{1,3}\s+", "", line)
            content = content.replace("**", "")
            yield (f"heading{level}", content)
        elif re.match(r"^\|[\s\-:|]+\|$", line):
            continue  # skip separator rows
        elif line.startswith("|") and line.endswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
            yield ("table_row", cells)
        elif re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line).replace("**", "").replace("*", "")
            yield ("bullet", content)
        elif re.match(r"^\s{2,}[-*]\s+", raw_line):
            content = re.sub(r"^[-*]\s+", "", line).replace("**", "").replace("*", "")
            yield ("sub_bullet", content)
        else:
            content = line.replace("**", "").replace("*", "")
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
        pdf.set_font("Helvetica", "B", 9)
        for i, row in enumerate(table_buffer):
            row = row + [""] * (n_cols - len(row))
            if i == 1 and all(set(c) <= {"-", ":", ""} for c in row):
                continue
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
        pdf.ln(3)

    for kind, content in parse_markdown_lines(text):
        if kind != "table_row":
            flush_table()

        if kind == "blank":
            pdf.ln(3)
        elif kind == "heading1":
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(max_width, 8, content)
            pdf.ln(2)
        elif kind == "heading2":
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(max_width, 7, content)
            pdf.ln(2)
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

    flush_table()
    return bytes(pdf.output())


pdf_bytes = markdown_to_pdf(markdown_content)

st.download_button(
    label="⬇️ Download as PDF",
    data=pdf_bytes,
    file_name=f"report_{selected}.pdf",
    mime="application/pdf",
)
