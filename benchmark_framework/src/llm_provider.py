from dotenv import load_dotenv
import dspy
import os

load_dotenv() 

def get_api_key(model_name: str) -> str:
    #TODO setup the API key access
    raise NotImplementedError("Function not implemented yet.")

def get_api_base(model_name: str) -> str:
    raise NotImplementedError("Function not implemented yet.")

def get_model_id(model_name: str) -> str:
    raise NotImplementedError("Function not implemented yet.")

def setup_llm_provider(model_name: str):

    if model_name not in ["qwen-2-5-72b", "mixtral-8x22b", "llama-3-3-70b", "gpt-oss-120b", "DeepSeek-V3"]:
        raise ValueError(f"Unsupported LLM: {model_name}")

    api_key = get_api_key(model_name)
    model_id = get_model_id(model_name)
    api_base = get_api_base(model_name)
    dspy_llm = dspy.LM(
        model=model_id,
        cache=True,
        max_tokens=8000,
        temperature=0,
        api_base=api_base,
        api_key=api_key,
    )
    dspy.configure(lm=dspy_llm)
    dspy.settings.configure(async_max_workers=50)