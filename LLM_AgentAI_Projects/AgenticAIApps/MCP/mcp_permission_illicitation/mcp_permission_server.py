import json
from pathlib import Path
from datetime import datetime
from fastmcp import FastMCP
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).parent / "data"
BASE_DIR.mkdir(exist_ok=True)

mcp = FastMCP("Permission-Aware MCP Server")
@mcp.tool()
def read_file(filepath: str) -> str:
    """
    Read a file from the data directory. (Risk: LOW)

    Args:
        filepath: Path to the file relative to data directory
    """
    try:
        file_path = BASE_DIR / filepath

        if not file_path.exists():
            return f"Error: File {filepath} not found"

        return file_path.read_text()
    except Exception as e:
        return f"Error reading file: {str(e)}"
@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file in the data directory. (Risk: MEDIUM)

    This operation modifies data and should require user confirmation via elicitation.

    Args:
        filepath: Path to the file relative to data directory
        content: Content to write to the file
    """
    try:
        file_path = BASE_DIR / filepath
        file_path.write_text(content)

        # Log the operation
        log_entry = f"[{datetime.now().isoformat()}] WRITE: {filepath}\n"
        audit_log = BASE_DIR / "audit.log"
        with open(audit_log, "a") as f:
            f.write(log_entry)

        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
@mcp.tool()
def delete_file(filepath: str) -> str:
    """
    Delete a file from the data directory. (Risk: HIGH)

    This operation is destructive and should require detailed user confirmation via elicitation.

    Args:
        filepath: Path to the file to delete
    """
    try:
        file_path = BASE_DIR / filepath

        if not file_path.exists():
            return f"Error: File {filepath} not found"

        file_path.unlink()

        # Log the operation
        log_entry = f"[{datetime.now().isoformat()}] DELETE: {filepath}\n"
        audit_log = BASE_DIR / "audit.log"
        with open(audit_log, "a") as f:
            f.write(log_entry)

        return f"Successfully deleted {filepath}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"
@mcp.tool()
def execute_command(command: str) -> str:
    """
    Execute a system command. (Risk: CRITICAL)

    This operation can affect system state and should require extensive user confirmation.
    For security, this is a simulation only.

    Args:
        command: The command to execute (simulated)
    """
    # Simulate command execution without actually running it
    log_entry = f"[{datetime.now().isoformat()}] EXECUTE (simulated): {command}\n"
    audit_log = BASE_DIR / "audit.log"
    with open(audit_log, "a") as f:
        f.write(log_entry)

    return f"Simulated execution of command: {command}\n(Actual execution disabled for security)"
@mcp.resource("file://audit/log")
def get_audit_log() -> str:
    """Get the audit log of all operations."""
    audit_log = BASE_DIR / "audit.log"
    if not audit_log.exists():
        return "No audit log entries yet."
    return audit_log.read_text()


@mcp.resource("file://config/permissions")
def get_permissions_config() -> str:
    """Get the current permissions configuration."""
    permissions_file = BASE_DIR / "permissions.json"
    if not permissions_file.exists():
        return json.dumps({
            "read_file": "allow",
            "write_file": "ask",
            "delete_file": "deny",
            "execute_command": "deny"
        }, indent=2)
    return permissions_file.read_text()
@mcp.prompt()
def security_review(operation: str, risk_level: str) -> list[dict]:
    """
    Generate a security review prompt for an operation.

    Args:
        operation: The operation to review
        risk_level: The risk level (LOW, MEDIUM, HIGH, CRITICAL)
    """
    return [
        {
            "role": "user",
            "content": f"""Review this operation for security implications:

Operation: {operation}
Risk Level: {risk_level}

Please analyze:
1. What data or systems could be affected?
2. What are the potential security risks?
3. What safeguards should be in place?
4. Should this operation require user approval?
5. What should be logged for audit purposes?
"""
        }
    ]
if __name__ == "__main__":
    mcp.run(transport="stdio")
