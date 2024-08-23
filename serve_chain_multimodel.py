#!/usr/bin/env python
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    # the default LLM ^
    default_key="gemini-1.5-flash",
    # This adds a new option, with name `openai` that is equal to `ChatOpenAI()`
    openai=ChatOpenAI(),
    # This adds a new option, with name `gpt4` that is equal to `ChatOpenAI(model="gpt-4")`
    gpt4=ChatOpenAI(model="gpt-4"),
    # You can add more configuration options here
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
    chain,
    path="/gemini-what",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
