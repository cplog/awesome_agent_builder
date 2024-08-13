
from langgraph.graph import END, StateGraph
from typing import Annotated, Dict, TypedDict

# Standard library imports
import os
import re
import sys

# Related third party imports
from dotenv import load_dotenv, find_dotenv

# Local application/library specific imports
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)

from langchain.schema.output_parser import StrOutputParser
import logging

# Load environment variables
load_dotenv(find_dotenv())
CURPATH = os.path.abspath(os.path.dirname(__file__))
mainPath = os.path.join(CURPATH.split("app")[0], 'app')
sys.path.append(mainPath)
os.chdir(mainPath)
from utils.database.io_mongo import init_mongo
from llm.chains.neo4j_runnable import GRAPH_QUERY_CHAIN

from utils.models import LLMModel, chat_model
# Define your desired data structure.
from typing import Dict, List, Optional, Union
from configs.schema.category import CategoryEnum, SubCategoryEnum
from configs.schema.color import ColorEnum

from pydantic import BaseModel, Field
from enum import Enum

from configs.schema.garment_class import Garment
# Langchain and prompt setup
import nest_asyncio
nest_asyncio.apply()

class GraphState(TypedDict):
    keys: Dict[str, any]
    question: Optional[str] = None
    classification: Optional[str] = None
    
    
workflow = StateGraph(GraphState)

llm35 = LLMModel('OPENAI')
llm35_model = chat_model(model=llm35, temperature=0.15)

# Define the classification function
def classify_input_func(question):
    # question = 'hi'
    prompt = ChatPromptTemplate.from_template("Classify the following as [greeting, search]: {question}")
    output_parser = StrOutputParser()
    chain = prompt | llm35_model | output_parser
    classification = chain.invoke({'question': question}).strip().lower()
    return "greeting" if classification == "greeting" else "search"

# Define the classification node using the LLM chain
def classify_input_node(state):
    print("Classifying input")
    question = state.get('question', '').strip()
    classification = classify_input_func(question)
    print(classification)
    return {"classification": classification}

# Add the classification node to the graph
workflow.add_node("classify_input", classify_input_node)


def handle_greeting_node(state):
    print("Handling greeting")
    return {"keys": {"response": "Hello! How can I help you today?"}}


def handle_search_node(state):
    print("Handling searching")
    question = state.get('question', '').strip()
    # Implement your search logic here
    search_result = "Search result for '{}'".format(question)
    return {"keys": {"response": search_result}}

workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_search", handle_search_node)

def decide_next_node(state):
    classification = state.get('classification', '')
    print(state)
    return "handle_greeting" if classification == "greeting" else "handle_search"

'''
Conditional edge mapping: This is a dictionary where the keys are the possible strings returned by the condition_function, and the values are the names of the nodes that the graph should transition to for each condition.
'''

workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_search": "handle_search"
    }
)

workflow.set_entry_point("classify_input")
workflow.add_edge('handle_greeting', END)
workflow.add_edge('handle_search', END)

app = workflow.compile()

inputs = {"question": "hi"}
result = app.invoke(inputs)
print(result)



