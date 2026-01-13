import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import toml
import requests
import argparse
from argparse import Namespace
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="D&D Assistant Answer Server")

config = None


class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file located in root folder."""
    return toml.load("src/config.toml")


def get_documents_from_retriever(question: str, retriever_url: str) -> str:
    """Get relevant documents from retriever server via HTTP."""
    try:
        response = requests.post(
            f"{retriever_url}/invoke", json={"input": question}, timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "output" in result:
            docs = result["output"]
            if isinstance(docs, list):
                context = "\n\n".join(
                    [
                        doc.get("page_content", "")
                        if isinstance(doc, dict)
                        else str(doc)
                        for doc in docs
                    ]
                )
                return context
            else:
                return str(docs)
        else:
            logger.warning(f"Unexpected retriever response format: {result}")
            return str(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling retriever server: {e}")
        raise


def format_prompt(context: str, question: str, prompt_template: str) -> str:
    """Format prompt template with context and question."""
    return prompt_template.format(context=context, question=question)


def call_vllm(
    prompt: str, temperature: float, top_p: float, max_tokens: int, vllm_url: str
) -> str:
    """Call vLLM server to generate answer."""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(vllm_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "")
        elif "text" in result:
            return result["text"]
        else:
            logger.warning(f"Unexpected vLLM response format: {result}")
            return str(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling vLLM server: {e}")
        raise


@app.post("/get_answer")
def get_answer(request_data: QuestionRequest):
    """Endpoint to get answer for a question."""
    global config

    try:
        temperature = (
            request_data.temperature
            if request_data.temperature is not None
            else config["vllm"]["default_temperature"]
        )
        top_p = (
            request_data.top_p
            if request_data.top_p is not None
            else config["vllm"]["default_top_p"]
        )
        max_tokens = (
            request_data.max_tokens
            if request_data.max_tokens is not None
            else config["vllm"]["default_max_tokens"]
        )

        logger.info(f"Retrieving documents for question: {request_data.question}")
        retriever_url = config["retriever"].get("url", "http://localhost:8000")
        context = get_documents_from_retriever(request_data.question, retriever_url)

        prompt = format_prompt(
            context=context,
            question=request_data.question,
            prompt_template=config["vllm"]["prompt_template"],
        )

        logger.info(f"Calling vLLM server at {config['vllm']['url']}")
        answer = call_vllm(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            vllm_url=config["vllm"]["url"],
        )

        return {
            "answer": answer,
            "question": request_data.question,
            "context": context,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Error in get_answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI answer server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host for the FastAPI server"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for the FastAPI server"
    )
    return parser.parse_args()


def main() -> None:
    global config

    args = parse_args()
    config = load_config()

    retriever_url = config["retriever"].get("url", "http://localhost:8000")
    logger.info(f"Retriever server URL: {retriever_url}")

    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
