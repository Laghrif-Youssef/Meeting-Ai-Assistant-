from fastmcp import FastMCP

from gmail_api import get_latest_emails, send_email, send_email_with_attachment


mcp = FastMCP(name="Gmail_MCP_Server")


@mcp.tool()
def send_email_tool(
    to: str,
    subject: str,
    body: str
) -> str:

    return send_email(
        to,
        subject,
        body
    )


@mcp.tool()
def send_email_with_attachment_tool(
    to: str,
    subject: str,
    body: str,
    file_paths: list[str]
) -> str:

    return send_email_with_attachment(
        to,
        subject,
        body,
        file_paths
    )


@mcp.tool()
def get_latest_emails_tool(
    n: int
) -> list:

    return get_latest_emails(n)


if __name__ == "__main__":

    mcp.run()