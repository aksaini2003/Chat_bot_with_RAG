from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from ddgs import DDGS
import streamlit as st 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# NOTE: The above code is for the proper working of the following code 

# -------------------
# 1. LLM + embeddings
# -------------------
# llm = ChatOpenAI(model="gpt-4o-mini")
groq_api_key=st.secrets['GROQ_API_KEY']
llm=ChatGroq(model="Llama-3.3-70b-Versatile",api_key=groq_api_key)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")\
# google_api_key=os.environ.get('GOOGLE_API_KEY')
google_api_key=st.secrets['GOOGLE_API_KEY']
embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',api_key=google_api_key)
# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id:
        thread_id_str = str(thread_id)
        if thread_id_str in _THREAD_RETRIEVERS:
            return _THREAD_RETRIEVERS[thread_id_str]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
# search_tool = DuckDuckGoSearchRun(region="us-en") UPDATING its with newer version 

#lets create a function for this work 
@tool
def search_tool(input: str):
    #here it will takes the text as input and return the list of output 
    """This is a search tool take a text as input and return the list of text from the real time websearch's data."""
    ddgs=DDGS() 
    output=ddgs.text(input,region='us-en',max_results=5) 
    return output  


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic math operations. Use this tool to calculate:
    - add: addition
    - sub: subtraction
    - mul: multiplication
    - div: division
    Returns the result of the operation.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    # stock_api_key=os.environ.get('STOCK_API_KEY')
    # for streamlit
    stock_api_key=st.secrets['STOCK_API_KEY']
    """
    Get the latest stock price for a company symbol.
    Provide the stock symbol (e.g., AAPL, TSLA, GOOGL).
    Returns current price and trading information.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api_key}"
    )
    r = requests.get(url)
    return r.json()
@tool 
def get_weather(location: str)-> dict:
    """"Fetch the current weather for a given city using the openweather api """
    # api_key=os.environ['WEATHER_API_KEY']
    api_key=st.secrets['WEATHER_API_KEY']
    url = f"https://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no"  # Optional: disables air quality data
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data


tools = [search_tool, get_stock_price, calculator, get_weather]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Create a custom rag_tool with thread_id bound
    @tool
    def rag_tool_with_thread(query: str) -> dict:
        """
        Search and retrieve relevant information from the uploaded PDF document.
        Use this tool to find answers to questions about the document content.
        """
        retriever = _get_retriever(thread_id if thread_id else None)
        if retriever is None:
            return {
                "error": "No document indexed for this chat. Upload a PDF first.",
                "query": query,
            }

        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA.get(str(thread_id) if thread_id else "", {}).get("filename"),
        }

    # Create tools list with the thread-bound rag tool
    tools_with_thread = [search_tool, get_stock_price, calculator, rag_tool_with_thread, get_weather]
    llm_with_thread_tools = llm.bind_tools(tools_with_thread)

    system_message = SystemMessage(
        content=("When answering questions:\n"
            "- Use the rag_tool_with_thread to search uploaded PDF documents\n"
            "- Use calculator for math problems\n"
            "- Use get_stock_price for stock information\n"
            "- Use web search for general information\n"
            "Keep responses concise and helpful."
        )
    )

    messages = [system_message, *state["messages"]]
    try:
        response = llm_with_thread_tools.invoke(messages, config=config)
    except Exception as e:
        # Fallback if tool calling fails
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=f"I encountered an error while processing your request. Please try again.")]}
    
    return {"messages": [response]}


def tool_node(state: ChatState, config=None):
    """Execute tools with thread_id context."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    
    # Create rag_tool with thread context
    @tool
    def rag_tool_with_thread(query: str) -> dict:
        """
        Search and retrieve relevant information from the uploaded PDF document.
        Use this tool to find answers to questions about the document content.
        """
        retriever = _get_retriever(thread_id if thread_id else None)
        if retriever is None:
            return {
                "error": "No document indexed for this chat. Upload a PDF first.",
                "query": query,
            }

        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA.get(str(thread_id) if thread_id else "", {}).get("filename"),
        }
    
    # Create tool node with thread-aware rag tool
    tools_with_thread = [search_tool, get_stock_price, calculator, rag_tool_with_thread, get_weather]
    tool_node_executor = ToolNode(tools_with_thread)
    
    return tool_node_executor.invoke(state, config=config)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def get_indexed_documents(thread_id: str) -> dict:
    """Get info about indexed documents for a thread."""
    thread_id_str = str(thread_id)
    return {
        "thread_id": thread_id_str,
        "has_document": thread_id_str in _THREAD_RETRIEVERS,
        "metadata": _THREAD_METADATA.get(thread_id_str, {}),
        "all_indexed_threads": list(_THREAD_RETRIEVERS.keys())
    }
