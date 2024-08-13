# Building Tool-Calling Conversational AI with LangChain and LangGraph: A Beginner's Guide

The world of artificial intelligence is evolving rapidly, making the creation of conversational AI systems more accessible than ever. Two powerful tools revolutionizing this field are **LangChain** and **LangGraph**. In this guide, we'll explore how these technologies can be combined to build a sophisticated AI assistant capable of handling complex conversations and tasks.

> **Note:** If you're unfamiliar with the basics of LangGraph, start with this article: [Introduction to LangGraph: A Beginner's Guide](https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141).

### Getting Started

Before we dive in, ensure you've installed the necessary packages and set up Groq LLM, which is free to use. Instructions can be found [here](https://python.langchain.com/v0.2/docs/integrations/chat/groq/). Groq works well for tool calling, and Llama 3 provides good results for testing this feature.

## What Are LangChain and LangGraph?

- **LangChain**: A framework for developing language model-powered applications, offering tools to enhance language models' capabilities and ease of use.
- **LangGraph**: A library for constructing stateful, multi-actor applications with LangChain, enabling the creation of complex workflows and decision trees for AI-driven systems.

## Key Concept: Tool Calling

**Tool calling** is a standout feature in LangChain, allowing the AI to interact with external systems or perform specific tasks via the `@tool` decorator. This adds significant flexibility, making the language model more agentic and capable.

### Important Requirements:

- **Function Type**: Must be of type `(str) -> str`, accepting a string as input and returning a string.
- **Docstring**: Each tool function must include a docstring describing its purpose.

### Example Tool Definitions:

```python
@tool
def fetch_stock_price(symbol: str):
    """Fetch the current stock price for a given symbol."""
    price = round(random.uniform(1, 1000), 2)
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
```

These tools can be seamlessly integrated into the AI’s decision-making process, allowing it to fetch real-time data or perform actions based on user requests.

## Building the AI Assistant

At the core of our system is the `Assistant` class, which encapsulates the AI's behavior and makes integration into more complex systems straightforward:

```python
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
```

### What This Class Does:
1. **Process the conversation state**: The AI can take in the current state and user inputs.
2. **Generate responses or call tools**: Based on the input, it decides whether to generate a response or use a tool.
3. **Handle empty responses**: If the model returns an empty response, it prompts for a meaningful output.

## Creating a Dynamic Conversation Flow with LangGraph

LangGraph allows us to define dynamic conversation flows through a graph structure:

```python
def build_graph():
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(tools_to_use))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder
```

### How It Works:
1. **Start with the assistant**: The conversation begins with the assistant generating a response or determining the need for a tool.
2. **Transition to tools**: If tool usage is required, the flow moves to the tools node.
3. **Return to the assistant**: Once the tool completes its task, control returns to the assistant, allowing the conversation to continue.

## Putting It All Together

Now, let's create our graph and start a conversation:

```python
if __name__ == "__main__":
    builder = build_graph()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "user_id": "cplog",
            "thread_id": thread_id,
        }
    }

    _printed = set()
    events = graph.stream(
        {"messages": ("user", 'what can you do?')}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
```

This code sets up the graph, initiates a conversation, and processes the events, printing the results.

## Flexibility in Agent Design

While the `Assistant` class presented here is one approach, the flexibility of tool calling and LangGraph allows for a wide range of designs. Depending on your needs, you might:
- Implement complex decision-making within the assistant.
- Add tools for various domains.
- Create multi-step reasoning processes.
- Incorporate memory or context management systems.

## Conclusion

With LangChain and LangGraph, you can build a powerful, flexible AI assistant capable of handling complex tasks and conversations. Tool calling significantly enhances the AI's capabilities by enabling interaction with external systems. This system is easily extendable, making it an excellent starting point for advanced AI applications.

Remember, this example is just one way to design an AI agent. The true strength of this approach lies in its adaptability – tailor it to meet the specific needs of your application.

Happy coding, and enjoy exploring the exciting world of AI development with LangChain and LangGraph!

For reference, the complete script of the tutorial can be found here:
[agent_tool_langgraph.py](https://github.com/cplog/awesome_agent_builder/blob/main/langchain_tutorials/1_agent_tool_langgraph.md)

The agent-building method is referenced from the [Customer Support Bot Tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/). For a more advanced structure, consider reading the full tutorial.
