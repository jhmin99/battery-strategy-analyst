from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


def chunk_text(line: str, size: int = 70) -> list[str]:
    if not line:
        return [" "]
    return [line[idx : idx + size] for idx in range(0, len(line), size)]


def main() -> None:
    md_path = Path("/Users/baekkangmin/ai_agent/reports/battery_strategy_report.md")
    pdf_path = Path(
        "/Users/baekkangmin/ai_agent/reports/battery_strategy_report.pdf"
    )
    font_path = Path(
        "/Users/baekkangmin/Desktop/SKALA/rag/ai-service/langgraph-v1/30-Project/data/fonts/NanumGothic.ttf"
    )

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.add_font("Korean", "", str(font_path))
    pdf.set_font("Korean", size=11)

    for raw_line in text.splitlines():
        for part in chunk_text(raw_line, 70):
            pdf.cell(0, 6, part, new_x="LMARGIN", new_y="NEXT")
        if raw_line.strip() == "":
            pdf.cell(0, 4, " ", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()
