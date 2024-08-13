# Imports
from datetime import datetime
import random
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Initialize ChatGroq
llm = ChatGroq(temperature=0, model_name="llama3-groq-70b-8192-tool-use-preview")

# Define State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

"""
The Assistant class is designed this way for several reasons:
It encapsulates the behavior of the AI assistant as a callable object.
It takes a runnable as an argument, which allows for flexibility in what kind of AI model or chain is used.
The __call__ method makes the class instances directly callable, which is useful for integration with LangGraph.
It handles the logic for re-prompting if the AI returns an empty response, ensuring more robust interactions.
The class processes the input state, which contains the conversation history, and produces a response or tool call. This design allows for easy integration into the graph structure and provides a clean interface for the AI's behavior.
"""
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_info": user_id}
            result = self.runnable.invoke(state)
            # Re-prompt if LLM returns an empty response
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the provided tools to assist with tasks such as fetching stock prices, getting current weather, and calculating age."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Define the beginner-friendly tools
@tool
def fetch_stock_price(symbol: str):
    """Fetch the current stock price for a given symbol."""
    # Simulate API call
    # Generate a realistic price (between $1 and $1000)
    base_price = random.uniform(1, 1000)
    fluctuation = random.uniform(-0.05, 0.05)  # -5% to +5%
    price = round(base_price * (1 + fluctuation), 2)
    return f"The current price of {symbol} is ${price:.2f}"

@tool
def get_current_weather(location: str):
    """Get the current weather for a given location."""
    weather_conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorm", "Snow", "Fog"]
    temperature = random.randint(-10, 40)
    condition = random.choice(weather_conditions)
    humidity = random.randint(30, 90)
    wind_speed = random.randint(0, 30)
    
    return f"Current weather in {location}:\n" \
            f"- Condition: {condition}\n" \
            f"- Temperature: {temperature}°C\n" \
            f"- Humidity: {humidity}%\n" \
            f"- Wind Speed: {wind_speed} km/h"

tools_to_use = [fetch_stock_price, get_current_weather]

# Prompts
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided tools to assist with tasks such as fetching stock prices, getting current weather, and calculating age."
              "\n\nCurrent user:\n\n{user_info}\n"
              "\nCurrent time: {time}."),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now())

# Runnables
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools_to_use)

# Helper functions
def handle_tool_error(error):
    return f"An error occurred while using the tool: {str(error)}"

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

"""
This graph structure allows for a dynamic conversation flow:
1. The assistant generates a response or decides to use a tool.
If a tool is needed, the flow moves to the tools node.
After tool use, control returns to the assistant.
This cycle can repeat as needed throughout the conversation.
The design is powerful because it:
Separates concerns: The assistant logic and tool execution are in different nodes.
Allows for easy addition of new tools or modification of the assistant's behavior.
Provides a clear, visual way to understand and modify the conversation flow.
Enables complex interactions, like chained tool use or multi-step reasoning, without complicating the main assistant logic.
"""
# Graph construction
def build_graph():
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(tools_to_use))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
            
    
if __name__ == "__main__":
    builder = build_graph()
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    # Add any additional execution code here
    import uuid
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # fetch the user's id
            "user_id": "cplog",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    _printed = set()
    events = graph.stream(
        {"messages": ("user", 'hi, who was born in 1992')}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
