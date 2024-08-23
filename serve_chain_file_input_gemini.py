#!/usr/bin/env python
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.pydantic_v1 import Field
from langchain_core.document_loaders import Blob
from langchain_core.runnables import RunnableLambda
from langserve import CustomUserType, add_routes
import base64

load_dotenv()

class FileProcessingRequest(CustomUserType):
    image_url: str = Field(..., extra={"widget": {"type": "base64file"}})


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

prompt_template =  HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": "Summarize this image"},
                {
                    "type": "image_url",
                    "image_url": "{encoded_image_url}",
                },
            ]
        )
summarize_image_prompt = ChatPromptTemplate.from_messages([prompt_template])
chain = summarize_image_prompt | llm | StrOutputParser()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="serving a simple chain",
)

add_routes(
    app,
    chain.with_types(input_type=FileProcessingRequest),
    path="/gemini-what",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
