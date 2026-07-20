import sys
import json
import gradio as gr
from mcp_permission_client_base import MCPPermissionClient


class MCPPermissionClientApp(MCPPermissionClient):
    """GUI client application with permission management interface."""

    def __init__(self, server_script: str):
        super().__init__(server_script)
        self.tools_cache = []
        self.prompts_cache = []
    async def gui_list_tools(self):
        """List tools with their permission status for GUI."""
        await self.connect()
        tools = await self.list_tools()

        output = "Available tools:\n\n"
        self.tools_cache = []

        for tool in tools:
            tool_name = tool.name
            permission = self.permissions.get(tool_name, "ask")
            self.tools_cache.append(tool_name)

            output += f"- {tool_name}\n"
            output += f"  Permission: {permission.upper()}\n"
            if tool.description:
                output += f"  Description: {tool.description}\n"
            output += "\n"

        choices = [f"{name} ({self.permissions.get(name, 'ask')})" for name in self.tools_cache]
        return output, gr.update(choices=choices)
    async def gui_call_tool(self, tool_selection: str, arguments_json: str, approved: bool = False):
        """Call a tool with permission checking for GUI."""
        if not tool_selection:
            return "Please select a tool first"

        # Extract tool name from selection (format: "tool_name (permission)")
        tool_name = tool_selection.split(" (")[0]

        # Parse arguments
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON in arguments: {str(e)}"

        # Call tool with permission check
        result = await self.call_tool_with_permission(tool_name, arguments, approved=approved)

        # Extract text content
        if isinstance(result, list) and len(result) > 0:
            content = result[0]
            if hasattr(content, 'text'):
                return content.text
            return str(content)

        return str(result)
    async def gui_list_resources(self):
        """List resources for GUI."""
        await self.connect()
        resources = await self.list_resources()

        output = "Available resources:\n\n"
        for resource in resources:
            output += f"- {resource.uri}\n"
            if resource.name:
                output += f"  Name: {resource.name}\n"
            if resource.description:
                output += f"  Description: {resource.description}\n"
            output += "\n"

        return output
    async def gui_read_resource(self, uri: str):
        """Read a resource for GUI."""
        if not uri.strip():
            return "Please enter a resource URI"

        await self.connect()
        contents = await self.read_resource(uri)

        if isinstance(contents, list) and len(contents) > 0:
            content = contents[0]
            if hasattr(content, 'text'):
                return content.text
            return str(content)

        return str(contents)
    async def gui_list_prompts(self):
        """List prompts for GUI."""
        await self.connect()
        prompts = await self.list_prompts()

        output = "Available prompts:\n\n"
        self.prompts_cache = []

        for prompt in prompts:
            self.prompts_cache.append(prompt.name)
            output += f"- {prompt.name}\n"
            if prompt.description:
                output += f"  Description: {prompt.description}\n"
            if hasattr(prompt, 'arguments') and prompt.arguments:
                args = [arg.name for arg in prompt.arguments]
                output += f"  Arguments: {', '.join(args)}\n"
            output += "\n"

        return output, gr.update(choices=self.prompts_cache)

    async def gui_get_prompt(self, prompt_name: str, arguments_json: str):
        """Get a rendered prompt for GUI."""
        if not prompt_name:
            return "Please select a prompt first"

        # Parse arguments
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON in arguments: {str(e)}"

        # Get prompt
        messages = await self.get_prompt(prompt_name, arguments)

        output = f"Prompt: {prompt_name}\n\n"
        for msg in messages:
            role = getattr(msg, 'role', 'unknown')
            content = getattr(msg, 'content', '')
            if hasattr(content, 'text'):
                content = content.text
            output += f"[{role}]: {content}\n\n"

        return output
    async def gui_configure_permission(self, tool_name: str, policy: str):
        """Configure permission for a tool."""
        if not tool_name:
            return "Please enter a tool name"

        if policy not in ["allow", "deny", "ask"]:
            return "Policy must be: allow, deny, or ask"

        self.permissions[tool_name] = policy
        self.save_permissions()

        return f"Permission updated: {tool_name} = {policy}\nPermissions saved to {self.permissions_file}"
    async def gui_view_audit_log(self):
        """View the audit log."""
        if not self.audit_log_file.exists():
            return "No audit log entries yet."

        return self.audit_log_file.read_text()
    def create_interface(self):
        """Create the Gradio interface with permission management."""

        with gr.Blocks(title="MCP Permission Client") as interface:
            gr.Markdown("""
            # MCP Permission Client
            Manage permissions, view audit logs, and interact with MCP tools securely.
            """)

            with gr.Tabs():
                with gr.Tab("Tools"):
                    gr.Markdown("### List and Call Tools with Permission Enforcement")
                    with gr.Row():
                        with gr.Column():
                            list_tools_btn = gr.Button("List Tools", variant="primary")
                            tools_output = gr.Textbox(label="Available Tools", lines=10)

                        with gr.Column():
                            tool_dropdown = gr.Dropdown(label="Select Tool", choices=[], interactive=True)
                            tool_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filepath": "test.txt"}',
                                lines=3
                            )
                            with gr.Row():
                                call_tool_btn = gr.Button("Call Tool", variant="primary")
                                approve_tool_btn = gr.Button("Approve & Execute", variant="secondary")
                            tool_result = gr.Textbox(label="Result", lines=10)

                    list_tools_btn.click(
                        fn=self.gui_list_tools,
                        outputs=[tools_output, tool_dropdown]
                    )

                    call_tool_btn.click(
                        fn=self.gui_call_tool,
                        inputs=[tool_dropdown, tool_args],
                        outputs=tool_result
                    )

                    async def gui_approve_tool(tool_selection, arguments_json):
                        return await self.gui_call_tool(tool_selection, arguments_json, approved=True)

                    approve_tool_btn.click(
                        fn=gui_approve_tool,
                        inputs=[tool_dropdown, tool_args],
                        outputs=tool_result
                    )

                with gr.Tab("Resources"):
                    gr.Markdown("### List and Read Resources")
                    with gr.Row():
                        with gr.Column():
                            list_resources_btn = gr.Button("List Resources", variant="primary")
                            resources_output = gr.Textbox(label="Available Resources", lines=10)

                        with gr.Column():
                            resource_uri = gr.Textbox(
                                label="Resource URI",
                                placeholder="file://audit/log"
                            )
                            read_resource_btn = gr.Button("Read Resource", variant="primary")
                            resource_content = gr.Textbox(label="Resource Content", lines=10)

                    list_resources_btn.click(
                        fn=self.gui_list_resources,
                        outputs=resources_output
                    )

                    read_resource_btn.click(
                        fn=self.gui_read_resource,
                        inputs=resource_uri,
                        outputs=resource_content
                    )

                with gr.Tab("Prompts"):
                    gr.Markdown("### List and Get Prompts")
                    with gr.Row():
                        with gr.Column():
                            list_prompts_btn = gr.Button("List Prompts", variant="primary")
                            prompts_output = gr.Textbox(label="Available Prompts", lines=5)

                        with gr.Column():
                            prompt_dropdown = gr.Dropdown(label="Select Prompt", choices=[], interactive=True)
                            prompt_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"operation": "write_file", "risk_level": "MEDIUM"}',
                                lines=2
                            )
                            get_prompt_btn = gr.Button("Get Prompt", variant="primary")
                            prompt_result = gr.Textbox(label="Prompt Messages", lines=10)

                    list_prompts_btn.click(
                        fn=self.gui_list_prompts,
                        outputs=[prompts_output, prompt_dropdown]
                    )

                    get_prompt_btn.click(
                        fn=self.gui_get_prompt,
                        inputs=[prompt_dropdown, prompt_args],
                        outputs=prompt_result
                    )

                with gr.Tab("Permissions"):
                    gr.Markdown("### Manage Permissions and View Audit Log")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Configure Tool Permission**")
                            list_tools_for_perm_btn = gr.Button("Load Tools", size="sm")
                            perm_tool_name = gr.Dropdown(
                                label="Tool Name",
                                choices=[],
                                allow_custom_value=True
                            )
                            perm_policy = gr.Radio(
                                choices=["allow", "deny", "ask"],
                                label="Permission Policy",
                                value="ask"
                            )
                            save_perm_btn = gr.Button("Save Permission", variant="primary")
                            perm_result = gr.Textbox(label="Result", lines=3)

                        with gr.Column():
                            gr.Markdown("**Audit Log**")
                            view_audit_btn = gr.Button("View Audit Log", variant="secondary")
                            audit_output = gr.Textbox(label="Audit Log", lines=15)

                    async def load_tools_for_dropdown():
                        tools = await self.list_tools()
                        tool_names = [tool.name for tool in tools]
                        return gr.Dropdown(choices=tool_names)

                    list_tools_for_perm_btn.click(
                        fn=load_tools_for_dropdown,
                        outputs=perm_tool_name
                    )

                    save_perm_btn.click(
                        fn=self.gui_configure_permission,
                        inputs=[perm_tool_name, perm_policy],
                        outputs=perm_result
                    )

                    view_audit_btn.click(
                        fn=self.gui_view_audit_log,
                        outputs=audit_output
                    )

        return interface
def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_permission_client_app.py <server_script>")
        print("Example: python mcp_permission_client_app.py mcp_permission_server.py")
        sys.exit(1)

    server_script = sys.argv[1]

    client = MCPPermissionClientApp(server_script)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7863)


if __name__ == "__main__":
    main()
