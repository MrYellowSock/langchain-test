#!/usr/bin/env python
"""Example LangChain server exposes multiple runnables (LLMs in this case)."""
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.9,
    ),
    path="/gemini",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
