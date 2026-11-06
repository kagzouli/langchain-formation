from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

load_dotenv()

from langchain.chat_models import init_chat_model

MAX_ITERATIONS=8
MODEL = "qwen3:4b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price in the catalog
    Args:
        product: A plain string containing the product name.
                 Examples: "laptop", "headphones", "keyboard".
                 Never provide a dictionary or an object.
    """

    print(f"  >>> Executing get_product_price(product='{product}')")
    prices=  { "laptop" : 1299.99, "headphones" : 129.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available: bronze, silver, gold.."""
    print(f" >>> Executing apply_discount(price={price}, discount_tier={discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount=discount_percentage.get(discount_tier, 0)
    return round(price* (1-discount/100), 2)


## Agent loop
def run_agent(question: str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"What is the question {question}")

    messages= [
        SystemMessage(
            content="You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool. \n\n "
            "STRICT RULES : You must follow exactly :\n"
            "1. Never guess or assume any product price."
            "You must call get_product_price to get the real price"
            "2. Only call apply_discount AFTER you have received"
            "a price from get_product_price. Pass the exact price."
            "return by get_product_price. do not pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math"
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier,"
            "ask them with tier to use. do NOT ASSUME one."
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n Interation {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tools_calls = ai_message.tool_calls

        # If not tools_calls
        if (not tools_calls):
            print(f"Final Answer : {ai_message.content}")
            return ai_message.content

        # Process only the first tool call
        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        print(f" [Tools selected] {tool_name} with tool args : {tool_args}")

        tool_use = tools_dict.get(tool_name)
        if (tool_use is None):
            raise ValueError(f"Tool : {tool_name} not found")

        observation = tool_use.invoke(tool_args)
        print(f" [Tool Result]: {observation}")

        # Add tool result to conversation history
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

if __name__ == '__main__':
    print("Hello Langchain Agent (.bind_tools)!")
    result = run_agent("What is the price of the laptop applying gold discount?")
    print(f"Result : {result}")


