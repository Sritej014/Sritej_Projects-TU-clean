import logging
from fastmcp import FastMCP
from pathlib import Path

# Suppress verbose FastMCP logging
logging.getLogger("fastmcp").setLevel(logging.WARNING)

mcp = FastMCP("lab-server")
BASE_DIR = Path.cwd()

@mcp.tool()
def echo(text: str) -> str:
    """Echo back the input text."""
    return f"Echo: {text}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    file_path = BASE_DIR / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote to {path}"

@mcp.resource("file://resources/{filename}")
def read_resource_file(filename: str) -> str:
    """Read a file from the resources directory."""
    file_path = BASE_DIR / "resources" / filename
    if not file_path.exists():
        return f"File not found: {filename}"
    return file_path.read_text(encoding="utf-8")

@mcp.prompt()
def review_file(filename: str) -> str:
    """Generate a prompt to review a file's contents."""
    return f"""Please review the file '{filename}' and provide:

1. A summary of its contents
2. Key points or sections
3. Any suggestions for improvement
4. Overall quality assessment

Use the appropriate tools to read the file if needed."""

if __name__ == "__main__":
    mcp.run()
