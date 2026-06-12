"""
pv_generator.py
---------------
Generates a professional PDF Procès-Verbal (meeting minutes) from structured PV data.
Uses reportlab Platypus for a clean, multi-section document.

Expected input (pv_data dict):
{
    "meeting_title":  "Réunion Commerciale Q2",
    "meeting_date":   "2026-06-25",
    "meeting_time":   "10:00",
    "location":       "Salle de conférence / Google Meet",
    "prepared_by":    "AssistantIA",
    "participants":   ["Alice Martin", "Bob Dupont", "Sara Alami"],
    "topics":         ["Point sur les ventes", "Stratégie Q3", "Budget marketing"],
    "decisions": [
        {
            "decision": "Augmenter le budget marketing de 15%",
            "context":  "Suite à la baisse des leads en mai"
        }
    ],
    "actions": [
        {
            "title":      "Préparer le rapport des ventes Q2",
            "responsible":"Alice Martin",
            "deadline":   "2026-07-01",
            "priority":   "HIGH",
            "kpi":        "Rapport livré avant le 1er juillet"
        }
    ],
    "summary": "Réunion productive. Tous les points ont été abordés.",
    "next_meeting": "2026-07-10 à 10h00"   # optional
}
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)


# ─────────────────────────────────────────────
# Brand colours
# ─────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1A3557")
MID_BLUE    = colors.HexColor("#2E6DA4")
LIGHT_BLUE  = colors.HexColor("#D6E8F7")
ACCENT      = colors.HexColor("#E8A020")
LIGHT_GRAY  = colors.HexColor("#F5F5F5")
MID_GRAY    = colors.HexColor("#CCCCCC")
TEXT_DARK   = colors.HexColor("#1E1E1E")

PRIORITY_COLORS = {
    "CRITICAL": colors.HexColor("#C0392B"),
    "HIGH":     colors.HexColor("#E67E22"),
    "MEDIUM":   colors.HexColor("#2980B9"),
    "LOW":      colors.HexColor("#27AE60"),
}


# ─────────────────────────────────────────────
# Style sheet
# ─────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["doc_title"] = ParagraphStyle(
        "doc_title",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["doc_subtitle"] = ParagraphStyle(
        "doc_subtitle",
        fontSize=11, fontName="Helvetica",
        textColor=colors.HexColor("#DDEEFF"), alignment=TA_CENTER,
        spaceAfter=2,
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading",
        fontSize=12, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_LEFT,
        leftIndent=8, spaceAfter=0, spaceBefore=0,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontSize=10, fontName="Helvetica",
        textColor=TEXT_DARK, spaceAfter=4, leading=14,
    )
    styles["body_bold"] = ParagraphStyle(
        "body_bold",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=TEXT_DARK, spaceAfter=4,
    )
    styles["label"] = ParagraphStyle(
        "label",
        fontSize=9, fontName="Helvetica-Bold",
        textColor=MID_BLUE, spaceAfter=2,
    )
    styles["small"] = ParagraphStyle(
        "small",
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#666666"),
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#999999"), alignment=TA_CENTER,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header",
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell",
        fontSize=9, fontName="Helvetica",
        textColor=TEXT_DARK, alignment=TA_LEFT, leading=12,
    )
    styles["table_cell_center"] = ParagraphStyle(
        "table_cell_center",
        fontSize=9, fontName="Helvetica",
        textColor=TEXT_DARK, alignment=TA_CENTER, leading=12,
    )
    return styles


# ─────────────────────────────────────────────
# Section header helper
# ─────────────────────────────────────────────

def _section_header(title: str, styles: dict) -> list:
    """Returns a coloured section header bar with the given title."""
    header_table = Table(
        [[Paragraph(f"  {title}", styles["section_heading"])]],
        colWidths=["100%"],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    return [Spacer(1, 0.35 * cm), header_table, Spacer(1, 0.2 * cm)]


# ─────────────────────────────────────────────
# Info card (meta table at the top)
# ─────────────────────────────────────────────

def _info_card(pv: dict, styles: dict) -> Table:
    rows = [
        ["Date",          pv.get("meeting_date", "—"),   "Heure",    pv.get("meeting_time", "—")],
        ["Lieu",          pv.get("location", "—"),        "Rédigé par", pv.get("prepared_by", "AssistantIA")],
        ["Participants",  ", ".join(pv.get("participants", [])), "", ""],
    ]

    col_widths = [2.8 * cm, 7 * cm, 2.8 * cm, 4.5 * cm]

    table_data = []
    for row in rows:
        table_data.append([
            Paragraph(str(row[0]), styles["label"]),
            Paragraph(str(row[1]), styles["body"]),
            Paragraph(str(row[2]), styles["label"]) if row[2] else "",
            Paragraph(str(row[3]), styles["body"])  if row[3] else "",
        ])

    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT_BLUE),
        ("BACKGROUND",    (2, 0), (2, -1), LIGHT_BLUE),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, MID_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("SPAN",          (1, 2), (3, 2)),   # participants spans remaining cols
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ─────────────────────────────────────────────
# Decisions section
# ─────────────────────────────────────────────

def _decisions_section(decisions: list, styles: dict) -> list:
    elements = _section_header("📋  Décisions prises", styles)

    if not decisions:
        elements.append(Paragraph("Aucune décision formelle enregistrée.", styles["body"]))
        return elements

    for i, d in enumerate(decisions, 1):
        block = []
        decision_text = d if isinstance(d, str) else d.get("decision", str(d))
        context_text  = d.get("context", "") if isinstance(d, dict) else ""

        block.append(Paragraph(
            f"<b>{i}.</b>  {decision_text}",
            styles["body"]
        ))
        if context_text:
            block.append(Paragraph(
                f"<i>Contexte : {context_text}</i>",
                ParagraphStyle("ctx", parent=styles["small"], leftIndent=18, spaceAfter=6)
            ))

        elements.append(KeepTogether(block))
        elements.append(Spacer(1, 0.1 * cm))

    return elements


# ─────────────────────────────────────────────
# Actions table
# ─────────────────────────────────────────────

def _actions_section(actions: list, styles: dict) -> list:
    elements = _section_header("✅  Plan d'actions", styles)

    if not actions:
        elements.append(Paragraph("Aucune action définie.", styles["body"]))
        return elements

    # Header row
    header = [
        Paragraph("#",           styles["table_header"]),
        Paragraph("Action",      styles["table_header"]),
        Paragraph("Responsable", styles["table_header"]),
        Paragraph("Échéance",    styles["table_header"]),
        Paragraph("Priorité",    styles["table_header"]),
        Paragraph("KPI",         styles["table_header"]),
    ]
    table_data = [header]

    for i, a in enumerate(actions, 1):
        if isinstance(a, str):
            # plain string action
            table_data.append([
                Paragraph(str(i), styles["table_cell_center"]),
                Paragraph(a,      styles["table_cell"]),
                Paragraph("—",    styles["table_cell_center"]),
                Paragraph("—",    styles["table_cell_center"]),
                Paragraph("—",    styles["table_cell_center"]),
                Paragraph("—",    styles["table_cell"]),
            ])
            continue

        priority = str(a.get("priority", "MEDIUM")).upper()
        p_color  = PRIORITY_COLORS.get(priority, MID_BLUE)

        priority_para = Paragraph(
            f'<font color="{p_color.hexval()}"><b>{priority}</b></font>',
            styles["table_cell_center"]
        )

        table_data.append([
            Paragraph(str(i),                          styles["table_cell_center"]),
            Paragraph(a.get("title", "—"),             styles["table_cell"]),
            Paragraph(a.get("responsible", "—"),       styles["table_cell_center"]),
            Paragraph(a.get("deadline", "—"),          styles["table_cell_center"]),
            priority_para,
            Paragraph(a.get("kpi", "—"),               styles["table_cell"]),
        ])

    col_widths = [0.7 * cm, 5.5 * cm, 3.2 * cm, 2.5 * cm, 2 * cm, 3.2 * cm]

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        # Alternating rows
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        # Borders
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, MID_GRAY),
        # Padding
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(t)
    return elements


# ─────────────────────────────────────────────
# Topics section
# ─────────────────────────────────────────────

def _topics_section(topics: list, styles: dict) -> list:
    elements = _section_header("🗣️  Sujets abordés", styles)

    if not topics:
        elements.append(Paragraph("Aucun sujet enregistré.", styles["body"]))
        return elements

    for i, topic in enumerate(topics, 1):
        elements.append(Paragraph(f"  {i}.  {topic}", styles["body"]))

    return elements


# ─────────────────────────────────────────────
# Summary section
# ─────────────────────────────────────────────

def _summary_section(summary: str, next_meeting: str | None, styles: dict) -> list:
    elements = _section_header("📝  Synthèse", styles)
    elements.append(Paragraph(summary or "Aucune synthèse disponible.", styles["body"]))

    if next_meeting:
        elements.append(Spacer(1, 0.2 * cm))
        elements.append(Paragraph(
            f"<b>Prochaine réunion :</b>  {next_meeting}",
            styles["body_bold"]
        ))

    return elements


# ─────────────────────────────────────────────
# Cover header (title banner)
# ─────────────────────────────────────────────

def _cover_header(pv: dict, styles: dict, page_width: float) -> Table:
    title    = pv.get("meeting_title", "Procès-Verbal de Réunion")
    subtitle = f"Procès-Verbal  •  {pv.get('meeting_date', '')}  •  {pv.get('meeting_time', '')}"

    content = [
        [Paragraph(title,    styles["doc_title"])],
        [Paragraph(subtitle, styles["doc_subtitle"])],
    ]

    usable = page_width - 4 * cm   # account for margins
    t = Table(content, colWidths=[usable])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return t


# ─────────────────────────────────────────────
# Page footer callback
# ─────────────────────────────────────────────

def _make_footer(pv: dict):
    def footer_cb(canvas, doc):
        canvas.saveState()
        footer_text = (
            f"Procès-Verbal — {pv.get('meeting_title', '')}  |  "
            f"{pv.get('meeting_date', '')}  |  "
            f"Page {doc.page}"
        )
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, footer_text)
        canvas.setStrokeColor(MID_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    return footer_cb


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def generate_pv_pdf(pv_data: dict, output_path: str) -> str:
    """
    Generate a professional PDF PV from pv_data dict.

    Args:
        pv_data:     structured meeting data (see module docstring for schema)
        output_path: full path where the PDF will be written

    Returns:
        output_path on success
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
        title=pv_data.get("meeting_title", "Procès-Verbal"),
        author=pv_data.get("prepared_by", "AssistantIA"),
    )

    styles   = _build_styles()
    page_w   = A4[0]
    story    = []

    # ── Cover header ──────────────────────────
    story.append(_cover_header(pv_data, styles, page_w))
    story.append(Spacer(1, 0.4 * cm))

    # ── Info card ─────────────────────────────
    story.append(_info_card(pv_data, styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── Topics ────────────────────────────────
    story.extend(_topics_section(pv_data.get("topics", []), styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── Decisions ─────────────────────────────
    story.extend(_decisions_section(pv_data.get("decisions", []), styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── Actions table ─────────────────────────
    story.extend(_actions_section(pv_data.get("actions", []), styles))
    story.append(Spacer(1, 0.3 * cm))

    # ── Summary ───────────────────────────────
    story.extend(_summary_section(
        pv_data.get("summary", ""),
        pv_data.get("next_meeting"),
        styles,
    ))

    # ── Signature block ───────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 0.2 * cm))
    generated_at = datetime.now().strftime("%d/%m/%Y à %H:%M")
    story.append(Paragraph(
        f"Document généré automatiquement par AssistantIA le {generated_at}",
        styles["footer"]
    ))

    doc.build(story, onFirstPage=_make_footer(pv_data), onLaterPages=_make_footer(pv_data))
    return output_path