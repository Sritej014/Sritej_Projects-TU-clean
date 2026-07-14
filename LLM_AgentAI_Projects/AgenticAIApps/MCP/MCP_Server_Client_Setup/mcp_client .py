import asyncio
import sys
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.history = []

    async def connect(self, server_script: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script],
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
        print("✓ Connected to MCP server")

    async def list_tools(self):
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        result = await self.session.call_tool(tool_name, arguments)
        self.history.append({"operation": "tool", "name": tool_name})  # Add this line
        return result

    async def list_resources(self):
        result = await self.session.list_resource_templates()
        return result.resourceTemplates

    async def read_resource(self, uri: str):
        result = await self.session.read_resource(uri)
        return result

    async def list_prompts(self):
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, arguments: dict):
        result = await self.session.get_prompt(prompt_name, arguments)
        return result


    async def run(self):
        print("\n=== MCP Client ===")
        print("Commands: tools | call | resources | read | prompts | prompt | help | quit\n")


        while True:
            cmd = input("> ").strip().lower()

            if cmd == "quit":
                break

            try:
                if cmd == "tools":
                    tools = await self.list_tools()
                    for t in tools:
                        print(f"  • {t.name}: {t.description}")

                elif cmd == "call":
                    tool_name = input("  Tool name: ").strip()
                    print("  Arguments (as JSON, for example, {\"text\": \"hello\"}): ")
                    args = json.loads(input("  ").strip())
                    result = await self.call_tool(tool_name, args)
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(f"  Result: {content.text}")
                                elif cmd == "history":
                    if self.history:
                        print("  Operation History:")
                        for i, entry in enumerate(self.history, 1):
                            print(f"    {i}. {entry['operation']}: {entry['name']}")
                    else:
                        print("  No operations yet")

                elif cmd == "resources":
                    resources = await self.list_resources()
                    if resources:
                        for r in resources:
                            name = getattr(r, 'name', getattr(r, 'description', 'Unnamed resource'))
                            uri_template = getattr(r, 'uriTemplate', getattr(r, 'uri', 'N/A'))
                            print(f"  • {name}")
                            print(f"    URI template: {uri_template}")
                    else:
                        print("  No resources available")

                elif cmd == "read":
                    uri = input("  URI: ").strip()
                    result = await self.read_resource(uri)
                    for content in result.contents:
                        if hasattr(content, 'text'):
                            print(f"\n{content.text}")

                elif cmd == "prompts":
                    prompts = await self.list_prompts()
                    for p in prompts:
                        args_info = ""
                        if p.arguments:
                            arg_names = [arg.name for arg in p.arguments]
                            args_info = f" (args: {', '.join(arg_names)})"
                        print(f"  • {p.name}: {p.description}{args_info}")

                elif cmd == "prompt":
                    prompt_name = input("  Prompt name: ").strip()
                    print("  Arguments (as JSON): ")
                    args = json.loads(input("  ").strip())
                    result = await self.get_prompt(prompt_name, args)
                    print(f"\n--- Prompt: {result.description} ---")
                    for msg in result.messages:
                        content_text = msg.content.text if hasattr(msg.content, 'text') else msg.content.get('text', '')
                        print(f"{msg.role}: {content_text}")
                elif cmd == "help":
                    print("""
                    Available Commands:
                    -------------------
                    tools       - List available tools
                    call        - Invoke a tool
                    resources   - List resource templates
                    read        - Read a resource by URI
                    prompts     - List prompt templates
                    prompt      - Get a rendered prompt
                    help        - Show this help message
                    quit        - Exit the client
                    """)

                else:
                    print("  Unknown command")

            except json.JSONDecodeError:
                print("  Error: Invalid JSON format")
                print("  Hint: Use double quotes, for example, {\"text\": \"hello\"}")
            except Exception as e:
                print(f"  Error: {e}")
                if "not found" in str(e).lower():
                    print("  Hint: Check the resource URI or filename")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect(sys.argv[1])
        await client.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
