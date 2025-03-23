# embedding.py
import logging
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

logger = logging.getLogger(__name__)

class OpenAIEmbeddings(BaseModel, Embeddings):
    """`OpenAI Embeddings` embedding models."""

    openai_api_key: Optional[str] = None
    """OpenAI application apikey"""
    model: str = "text-embedding-ada-002"
    """Model name to use for embedding"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate whether openai_api_key in the environment variables or
        configuration file are available or not.

        Args:
            values: a dictionary containing configuration information, must include the
            fields of openai_api_key
        Returns:
            a dictionary containing configuration information. If openai_api_key
            are not provided in the environment variables or configuration
            file, the original values will be returned; otherwise, values containing
            openai_api_key will be returned.
        Raises:
            ValueError: openai package not found, please install it with `pip install
            openai`
        """
        values["openai_api_key"] = get_from_dict_or_env(
            values,
            "openai_api_key",
            "OPENAI_API_KEY",
        )

        try:
            import openai
            openai.api_key = values["openai_api_key"]
        except Exception as e:
            raise ValueError(
                "OpenAI package not found or API key not set, please install it with "
                "`pip install openai` and set the OPENAI_API_KEY environment variable."
            ) from e
        return values

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.langchain_embeddings = LangchainOpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model=self.model
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embedding a text.

        Args:
            text (str): A text to be embedded.

        Returns:
            List[float]: An embedding list of input text, which is a list of floating-point values.
        """
        return self.langchain_embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                            Each embedding is represented as a list of float values.
        """
        return self.langchain_embeddings.embed_documents(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `aembed_query`. Official does not support asynchronous requests")