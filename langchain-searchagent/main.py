from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class Source(BaseModel):
    """Scheme by the source used by the agent """
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Scheme for the answer response"""
    answer:str = Field(description="Agent answer for the query")
    sources: List[Source] = Field(default_factory=list, description="list of sources to generate the answer")

load_dotenv()

@tool
def searchData(query: str) -> str:
    """
    Tool that search over internet

    :param query: *The query to search for
    :return result
    """
    print(f"Searching for {query}")
    result = ""
    match query:
        case "Paris weather":
            result = "Paris weather is sunny"
        case "London weather":
            result = "London weather is gray"
        case "Dublin weather":
            result =  "Dublin weather is raining"

    return result




llm = ChatOllama(temperature=0, model="llama3.2:3b")
tools = [searchData]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
def main():
    print("Hello from langchain-searchagent!")

    response = agent.invoke({"messages" : HumanMessage(content="What is the weather in Paris?") })
    print(response)


if __name__ == "__main__":
    main()
