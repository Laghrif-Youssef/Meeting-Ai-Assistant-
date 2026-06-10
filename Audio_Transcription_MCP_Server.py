from fastmcp import FastMCP
import ollama

from Transcriber import transcribe

mcp = FastMCP(
    name="Transcription_MCP_Server"
)

@mcp.tool()
def transcribe_tool(audio_file: str) -> dict:
    """
    Transcribe an audio recording into text.
    Supports mp3, wav, m4a and other common formats.
    """

    return transcribe(audio_file)


def generate_pv(transcript: str):

    prompt = f"""
Analyze the following meeting transcript.

Extract:

1. Participants
2. Main topics discussed
3. Decisions made
4. Action items
5. Responsible person for each action
6. Deadlines if mentioned

Transcript:

{transcript}
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]



if __name__ == "__main__":

    mcp.run()