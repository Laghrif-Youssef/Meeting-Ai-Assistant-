import whisper
import warnings


model = whisper.load_model("base")

def transcribe(audio_file: str) -> dict:

    warnings.filterwarnings("ignore")

    result = model.transcribe(audio_file)

    return {
        "success": True,
        "language": result["language"],
        "transcript": result["text"]
    }

SYSTEM_PROMPT = """
You are an AI assistant.

You ONLY have access to these tools:

1. send_email_tool
   - sends an email
   - arguments:
       to
       subject
       body

1. send_email_with_attachment_tool
   - sends an email with attachment
   - arguments:
       to
       subject
       body
       file_paths

2. get_latest_emails_tool
   - returns latest inbox emails
   - argument:
       n

NEVER invent tool names.

NEVER create tools that do not exist.

If the user asks:
- for latest email body
- latest subject
- latest sender

you MUST:
1. call get_latest_emails_tool
2. analyze the returned data
3. answer from the returned data

If you decide to use a tool:
- respond ONLY with JSON
- do NOT add explanations
- do NOT add markdown
- do NOT add text before JSON

Tool call format:
{
  "tool": "...",
  "arguments": { ... }
}

Otherwise answer normally.
"""
