from functools import lru_cache
from typing import List, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

from .constants import LLM_MODEL_ID, LLM_MAX_NEW_TOKENS, LLM_DO_TOKENS, HF_TOKEN


class MistralLLM(LLM):
    """Minimal LangChain LLM wrapper autour du modèle Mistral HF."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    do_sample: bool = LLM_DO_TOKENS

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "mistral_local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Méthode appelée par LangChain pour générer du texte."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # On essaie de retirer le prompt du début si le modèle le recopie
        if text.startswith(prompt):
            text = text[len(prompt):]

        # Gestion des tokens d’arrêt si LangChain en passe
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]

        return text.strip()


@lru_cache(maxsize=1)
def get_tokenizer_and_model(model_id: str = LLM_MODEL_ID):
    """
    Charge le modèle Mistral et le tokenizer.
    (cache pour ne charger qu'une fois par process)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    cfg = AutoConfig.from_pretrained(model_id, token=HF_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",   # sur ta GPU
        config=cfg,
        token=HF_TOKEN,
    )

    return tokenizer, model


@lru_cache(maxsize=1)
def get_llm(model_id: str = LLM_MODEL_ID) -> MistralLLM:
    """
    Construit un LLM LangChain à partir du modèle HF.
    """
    tokenizer, model = get_tokenizer_and_model(model_id)
    llm = MistralLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=LLM_DO_TOKENS,
    )
    return llm