from txtai import LLM
from app.config import LLM_MODEL
import streamlit as st

logger = st.logger.get_logger(__name__)

class TxtaiLLM:
    """
    Wrapper for txtai's LLM functionality.
    """
    def __init__(self, model_path: str = LLM_MODEL):
        self.model_path = model_path
        self.llm = None # Lazy load
        logger.info(f"TxtaiLLM configured with model path: {self.model_path}")

    def _load_llm(self):
        if self.llm is None:
            logger.info(f"Loading LLM model: {self.model_path}...")
            self.llm = LLM(self.model_path)
            logger.info("LLM model loaded.")

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Generates a response from the LLM for a single prompt.

        Args:
            prompt: The user's prompt.
            system_prompt: Optional system message.
            **kwargs: Additional arguments for the LLM (e.g., maxlength).

        Returns:
            The LLM's generated text.
        """
        self._load_llm()
        # txtai's LLM takes a list of message dicts or a string
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.llm(messages, **kwargs)
        return self.llm(prompt, **kwargs)

    def batch_generate(self, prompts: list, **kwargs) -> list[str]:
        """
        Generates responses for a batch of prompts.
        Each item in `prompts` can be a string or a list of message dicts.

        Args:
            prompts: A list of prompts (strings or message lists).
            **kwargs: Additional arguments for the LLM (e.g., maxlength, batch_size).

        Returns:
            A list of LLM's generated texts.
        """
        self._load_llm()
        return self.llm(prompts, **kwargs)

    def stream(self, prompt: str, system_prompt: str = None, **kwargs):
        """
        Generates a streaming response from the LLM.

        Args:
            prompt: The user's prompt.
            system_prompt: Optional system message.
            **kwargs: Additional arguments for the LLM.

        Yields:
            Tokens from the LLM.
        """
        self._load_llm()
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            yield from self.llm(messages, stream=True, **kwargs)
        else:
            yield from self.llm(prompt, stream=True, **kwargs)