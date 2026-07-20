import json
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPPermissionClient:
    """Base MCP client with permission checking and audit logging."""

    def __init__(self, server_script: str, permissions_file: str = "data/permissions.json"):
        self.server_script = server_script
        self.permissions_file = Path(permissions_file)
        self.permissions_file.parent.mkdir(exist_ok=True)
        self.audit_log_file = self.permissions_file.parent / "audit.log"
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._connected = False
        self.permissions = self.load_permissions()
    async def connect(self):
        """Connect to the MCP server via STDIO. Safe to call multiple times."""
        if self._connected:
            return

        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()
        self._connected = True
    def load_permissions(self) -> dict:
        """Load permissions from file or return defaults."""
        if self.permissions_file.exists():
            return json.loads(self.permissions_file.read_text())
        return {
            "read_file": "allow",
            "write_file": "ask",
            "delete_file": "deny",
            "execute_command": "deny"
        }

    def save_permissions(self):
        """Save current permissions to file."""
        self.permissions_file.write_text(json.dumps(self.permissions, indent=2))

    def check_permission(self, tool_name: str, arguments: dict) -> str:
        """
        Check permission for a tool call.

        Returns: "allow", "deny", or "ask"
        """
        # Check for argument-specific permission
        arg_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        if arg_key in self.permissions:
            return self.permissions[arg_key]

        # Check for general tool permission
        return self.permissions.get(tool_name, "ask")
    def log_audit(self, operation: str, decision: str, reason: str = ""):
        """Log an operation to the audit log."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {operation} - Decision: {decision}"
        if reason:
            log_entry += f" - Reason: {reason}"
        log_entry += "\n"

        with open(self.audit_log_file, "a") as f:
            f.write(log_entry)
    async def request_elicitation(self, schema: dict, description: str) -> dict:
        """
        Request structured user input via elicitation.

        This is a conceptual implementation. In production, this would
        trigger a UI dialog and wait for user response.

        Args:
            schema: JSON schema for the required input
            description: Human-readable description of what's needed

        Returns:
            Dictionary with user's input
        """
        # Conceptual: In real implementation, this would show a UI dialog
        # and block until user provides input or cancels
        print(f"\nElicitation requested: {description}")
        print(f"Schema: {json.dumps(schema, indent=2)}")
        print("(Conceptual - automatic approval for demo)")

        # For demo purposes, return empty dict (representing user approval)
        return {}
    async def list_tools(self):
        """List all available tools from the server."""
        await self.connect()
        result = await self.session.list_tools()
        return result.tools

    async def call_tool_with_permission(self, tool_name: str, arguments: dict = None, approved: bool = False):
        """Call a tool after checking permissions."""
        await self.connect()

        if arguments is None:
            arguments = {}

        # Check permission
        permission = self.check_permission(tool_name, arguments)

        if permission == "deny":
            self.log_audit(f"TOOL: {tool_name}", "DENIED", "Policy: deny")
            return [type('obj', (), {'text': f"Permission denied for tool: {tool_name}"})]

        if permission == "ask" and not approved:
            # Log that approval was requested
            self.log_audit(f"TOOL: {tool_name}", "ASK", "Awaiting approval")

            # Return approval request message
            approval_msg = f"""Permission required for tool: {tool_name}
Arguments: {json.dumps(arguments, indent=2)}

This tool requires approval before execution.
Please approve this operation in the GUI to proceed."""
            return [type('obj', (), {'text': approval_msg})]

        # Execute the tool
        self.log_audit(f"TOOL: {tool_name}", "ALLOWED", f"Policy: {permission}")
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result.content
    async def list_resources(self):
        """List all available resources from the server."""
        await self.connect()
        result = await self.session.list_resources()
        return result.resources

    async def read_resource(self, uri: str):
        """Read a resource by URI."""
        await self.connect()
        result = await self.session.read_resource(uri=uri)
        return result.contents
    async def list_prompts(self):
        """List all available prompts from the server."""
        await self.connect()
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, arguments: dict = None):
        """Get a rendered prompt template."""
        await self.connect()
        if arguments is None:
            arguments = {}
        result = await self.session.get_prompt(name=prompt_name, arguments=arguments)
        return result.messages
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
