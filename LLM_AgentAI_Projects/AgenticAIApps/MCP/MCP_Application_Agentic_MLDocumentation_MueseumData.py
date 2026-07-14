# Standard library imports
import asyncio
# Third-party imports for MCP (Model Context Protocol) and LangGraph
from langchain_mcp_adapters.client import MultiServerMCPClient  # Connects to MCP servers
from langgraph.prebuilt import create_react_agent                # Creates ReAct-style agents
from langgraph.checkpoint.memory import InMemorySaver           # Provides conversation memory
from langchain_openai import ChatOpenAI                         # OpenAI chat model integration

async def main():
    """
    Main function that sets up and runs an AI agent with access to multiple MCP servers.
    The agent can access Context7 library documentation and Met Museum collections.
    """
    # Configure MCP (Model Context Protocol) servers
    # These servers provide tools that the AI agent can use
    client = MultiServerMCPClient(
        {
            # Context7 server - provides access to library documentation
            "context7": {
                "url": "https://mcp.context7.com/mcp",        # Server endpoint
                "transport": "streamable_http",                # Communication protocol
            },
            # Met Museum server - provides access to museum collection data
            "met-museum": {
                "command": "npx",                              # Node.js package runner
                "args": ["-y", "metmuseum-mcp"],              # Install and run met museum MCP
                "transport": "stdio",                         # Communication via stdin/stdout
            }
        }
    )
    # Initialize the OpenAI language model
    # This model will power the AI agent's reasoning and responses
    openai_model = ChatOpenAI(
        model="gpt-5-nano",  # Using OpenAI's GPT-5 Nano model
    )
    # Retrieve all available tools from the configured MCP servers
    # These tools allow the agent to interact with external services
    tools = await client.get_tools()
    # Set up conversation memory using InMemorySaver
    # This allows the agent to remember previous messages in the conversation
    checkpointer = InMemorySaver()
    # Configuration for conversation persistence
    # The thread_id ensures all messages in this session are grouped together
    config = {"configurable": {"thread_id": "conversation_id"}}
    # Create the ReAct agent with all components
    # ReAct = Reasoning + Acting (agent can reason about and use tools)
    agent = create_react_agent(
        model=openai_model,         # The language model to use
        tools=tools,                # Available tools from MCP servers
        checkpointer=checkpointer   # Memory system for conversation history
    )
    # Send initial message to introduce the agent and its capabilities
    response = await agent.ainvoke(
        {"messages": [
            # System message defines the agent's role and personality
            {"role": "system", "content": "You are a smart, useful agent with tools to access code library documentation and the Met Museum collection."},
            # User message requests the agent to introduce itself
            {"role": "user", "content": "Give a brief introduction of what you do and the tools you can access."},
        ]},
        config=config  # Use the conversation thread for memory persistence
    )
    # Print the agent's response (last message in the conversation)
    print(response['messages'][-1].content)
    # Main interaction loop - allows continuous conversation with the agent
    while True:
        # Display menu options to the user
        choice = input("""
        Menu:

        Ask the agent a question

        Quit
        Enter your choice (1 or 2): """)
    if choice == "1":
         # Get user's question
        print("Your question")
        query = input("> ")
         # Send the user's question to the agent
         # The agent will have access to the full conversation history
        response = await agent.ainvoke(
             {"messages": query},        # User's current question
             config=config              # Maintains conversation thread
         )
         # Display the agent's response
        print(response['messages'][-1].content)
    else:
         # Exit the program for any choice other than "1"
        print("Goodbye!")


if __name__ == "__main__":
    # Use asyncio to run the async main function
    asyncio.run(main())
