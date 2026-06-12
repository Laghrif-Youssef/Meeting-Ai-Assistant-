"""
Audio Transcription MCP Server
Tools:
  - transcribe_tool      : transcribe an audio file → raw text
  - generate_pv_tool     : LLM extracts structured data from transcript → returns dict
  - generate_pv_pdf_tool : full pipeline → transcript + LLM extraction + PDF generation
                           returns the path to the saved PDF
"""

import os
import json
from datetime import datetime

import ollama
from fastmcp import FastMCP

from Transcriber import transcribe
from pv_generator import generate_pv_pdf

mcp = FastMCP(name="Transcription_MCP_Server")

# Where PV PDFs are saved — override with env var if needed
PV_OUTPUT_DIR = os.environ.get("PV_OUTPUT_DIR", "pv_output")

MODEL = "qwen2.5:3b"   # use same model as the rest of the project


# ─────────────────────────────────────────────
# Tool 1 — Transcribe audio
# ─────────────────────────────────────────────

@mcp.tool()
def transcribe_tool(audio_file: str) -> dict:
    """
    Transcribe an audio recording into raw text.
    Supports mp3, wav, m4a and other common audio formats.

    Args:
        audio_file: path to the audio file

    Returns:
        dict with keys: success, transcript, duration_seconds (if available)
    """
    return transcribe(audio_file)


# ─────────────────────────────────────────────
# Tool 2 — Extract structured PV data from transcript
# ─────────────────────────────────────────────

@mcp.tool()
def generate_pv_tool(
    transcript: str,
    meeting_title: str = "Réunion",
    meeting_date: str = "",
    meeting_time: str = "",
    location: str = "",
) -> dict:
    """
    Analyse a meeting transcript and extract structured PV data using the LLM.
    Returns a structured dict (participants, topics, decisions, actions, summary).
    Does NOT generate a PDF — use generate_pv_pdf_tool for that.

    Args:
        transcript:    raw meeting transcript text
        meeting_title: title to use in the PV header
        meeting_date:  meeting date string (e.g. "25/06/2026")
        meeting_time:  meeting time string (e.g. "10:00")
        location:      meeting location or platform
    """
    today = datetime.now().strftime("%d/%m/%Y")

    prompt = f"""
You are an expert meeting secretary. Read the transcript below and extract information.

STRICT RULES:
- participants: list ONLY names actually spoken in the transcript. If no names appear, return [].
- topics: list the main subjects discussed. Infer from context if not stated explicitly.
- decisions: list only REAL decisions made. A decision is a commitment or agreement, not a task.
  If no decisions were made, return [].
- actions: each action MUST have a responsible person named in the transcript.
  - title: what must be done (from the transcript)
  - responsible: the EXACT name mentioned (e.g. "Alice", "Bob", "Sarah")
  - deadline: ONLY if explicitly mentioned (e.g. "before July 1st" → "01/07/2026").
    If no deadline was mentioned, use "Non défini". NEVER invent a date.
  - priority: use CRITICAL only if the word "critical", "urgent", or "critique" was used.
    Otherwise use HIGH for important tasks, MEDIUM for normal tasks, LOW for minor ones.
  - kpi: a measurable success criterion. If none mentioned, use "".
- summary: 2-3 sentences summarizing the meeting outcome.

Return ONLY a valid JSON object — no markdown, no explanation:
{{
  "participants": [],
  "topics": [],
  "decisions": [],
  "actions": [],
  "summary": ""
}}

TRANSCRIPT:
{transcript}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_ctx": 8192},
    )

    raw = response["message"]["content"]

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw text wrapped in a minimal structure
        extracted = {
            "participants": [],
            "topics":       [],
            "decisions":    [],
            "actions":      [],
            "summary":      raw,
        }

    # Attach meeting metadata
    extracted["meeting_title"] = meeting_title
    extracted["meeting_date"]  = meeting_date or today
    extracted["meeting_time"]  = meeting_time
    extracted["location"]      = location
    extracted["prepared_by"]   = "AssistantIA"

    return extracted


# ─────────────────────────────────────────────
# Tool 3 — Full pipeline: audio → transcript → PV → PDF
# ─────────────────────────────────────────────

@mcp.tool()
def generate_pv_pdf_tool(
    audio_file: str,
    meeting_title: str = "Réunion",
    meeting_date: str = "",
    meeting_time: str = "",
    location: str = "",
    output_filename: str = "",
) -> dict:
    """
    Full pipeline: transcribe audio → extract structured PV data → generate PDF.

    Args:
        audio_file:      path to the audio recording (mp3, wav, m4a, ...)
        meeting_title:   title for the PV header
        meeting_date:    date of the meeting (e.g. "25/06/2026")
        meeting_time:    time of the meeting (e.g. "10:00")
        location:        location or platform (e.g. "Google Meet")
        output_filename: optional custom filename for the PDF (without extension)
                         defaults to "PV_<meeting_title>_<date>.pdf"

    Returns:
        dict with keys:
          - success (bool)
          - pdf_path (str)         : full path to the generated PDF
          - transcript (str)       : raw transcript text
          - pv_data (dict)         : structured PV extracted by the LLM
          - error (str, optional)  : error message if something failed
    """
    # ── Step 1: Transcribe ────────────────────
    try:
        transcription_result = transcribe(audio_file)
        transcript = (
            transcription_result.get("transcript", "")
            or transcription_result.get("text", "")
            or str(transcription_result)
        )
        if not transcript.strip():
            return {"success": False, "error": "Transcription returned empty text."}
    except Exception as e:
        return {"success": False, "error": f"Transcription failed: {e}"}

    # ── Step 2: Extract structured PV with LLM ─
    try:
        pv_data = generate_pv_tool(
            transcript=transcript,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
            meeting_time=meeting_time,
            location=location,
        )
    except Exception as e:
        return {"success": False, "error": f"PV extraction failed: {e}", "transcript": transcript}

    # ── Step 3: Build output path ─────────────
    os.makedirs(PV_OUTPUT_DIR, exist_ok=True)

    if output_filename:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in output_filename)
        pdf_filename = f"{safe_name}.pdf"
    else:
        date_slug = (meeting_date or datetime.now().strftime("%Y-%m-%d")).replace("/", "-")
        title_slug = "".join(c if c.isalnum() else "_" for c in meeting_title)[:40]
        pdf_filename = f"PV_{title_slug}_{date_slug}.pdf"

    pdf_path = os.path.join(PV_OUTPUT_DIR, pdf_filename)

    # ── Step 4: Generate PDF ──────────────────
    try:
        generate_pv_pdf(pv_data, pdf_path)
    except Exception as e:
        return {
            "success":    False,
            "error":      f"PDF generation failed: {e}",
            "transcript": transcript,
            "pv_data":    pv_data,
        }

    return {
        "success":    True,
        "pdf_path":   os.path.abspath(pdf_path),
        "transcript": transcript,
        "pv_data":    pv_data,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()