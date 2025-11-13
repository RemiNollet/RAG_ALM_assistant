from functools import lru_cache
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline

from .constants import LLM_MODEL_ID, LLM_MAX_NEW_TOKENS, LLM_DO_TOKENS, HF_TOKEN


@lru_cache(maxsize=1)
def get_tokenizer_and_model(model_id: str = LLM_MODEL_ID):
    """
    Load the Mistral model and tokenizer.
    Cached so we only load once per process.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    cfg = AutoConfig.from_pretrained(model_id, token=HF_TOKEN)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        config=cfg,
        quantization_config=bnb_config,
        token=HF_TOKEN
    )
    return tokenizer, model


@lru_cache(maxsize=1)
def get_llm(model_id: str = LLM_MODEL_ID) -> HuggingFacePipeline:
    """
    Build a LangChain LLM from a HuggingFace text-generation pipeline.
    """
    tokenizer, model = get_tokenizer_and_model(model_id)

    gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=LLM_DO_TOKENS,
        return_full_text=False,  # important: we don't want the prompt back
    )

    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return llm