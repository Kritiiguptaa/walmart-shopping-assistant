# LangChain_API_KEY=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
# OPENAI_API_KEY=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
# LangChain_Project="chatbot"

#LangChain is a powerful framework for building applications that use language models like GPT (OpenAI), Claude, or local LLMs.

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI




load_dotenv()

## Environment Variables Call
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#LangSmith tracking
os.environ["LangChain_API_KEY"] = os.getenv("LangChain_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

##CREATE CHATBOT

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Walmart assistant. You will answer questions using the product information provided."),
    ("human", "Question: {question}\n\nMatched Products:\n{product_data}")
])

# Streamlit Framework
st.title("Langchain Walmart Chatbot")
input_text=st.text_input("Ask a question about Walmart products:")

# Open AI LLM Call
# llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
# output_parser=StrOutputParser()
from langchain_community.llms import OpenAI

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mixtral-8x7b-instruct",
    temperature=0.2
)


output_parser = StrOutputParser()

#chain
chain=prompt | llm | output_parser
if input_text:
    response=chain.invoke({"question": input_text})
    st.write(response)