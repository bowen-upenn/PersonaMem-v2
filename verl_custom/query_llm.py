"""
LLM query functionality with action-based prompt dispatch.
"""

import os
import logging
import openai
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import asyncio
import time
from typing import Any, List, Dict, Optional

from tqdm.asyncio import tqdm_asyncio, tqdm


# Setup logging
logger = logging.getLogger(__name__)
# Suppress httpx INFO logs to reduce console noise
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMQueryEngine:
    """
    Shared class for querying LLMs (OpenAI/Azure) with configuration from environment variables.
    """
    
    def __init__(self, rate_limit_per_min: int = 20, use_embeddings: bool = False):
        """Initialize the LLM client based on environment configuration."""
        self.client = None
        self.model = None
        self.rate_limit_per_min = rate_limit_per_min
        self.request_times = []
        self.use_embeddings = use_embeddings
        self.semaphore = asyncio.Semaphore(rate_limit_per_min)  # Max number of concurrent requests

        # Load environment variables from parent directory with force override
        abs_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
        load_dotenv(abs_env_path, override=True)
            
        self._setup_client()


    def _setup_client(self):
        """Setup OpenAI or Azure OpenAI client based on environment variables."""
        # Check for Azure OpenAI configuration first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        if self.use_embeddings:
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBED")
        else:
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_RLVR")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        print(f"Debug - Environment variables:")
        print(f"  AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
        print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {azure_deployment}")
        print(f"  AZURE_OPENAI_API_VERSION: {azure_api_version}")

        if azure_endpoint and azure_key and azure_deployment:
            logger.info("Using Azure OpenAI configuration")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=azure_api_version,
            )
            self.model = azure_deployment
        else:
            # Fall back to OpenAI configuration
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL")
            
            if openai_api_key:
                logger.info("Using OpenAI configuration")
                self.client = OpenAI(api_key=openai_api_key)
                self.model = openai_model
            else:
                raise ValueError(
                    "No valid LLM configuration found. Please set either:\n"
                    "Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME\n"
                    "or OpenAI: OPENAI_API_KEY"
                )


    def get_sentence_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get sentence embeddings for a list of texts using the configured LLM client.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            List[List[float]]: List of embeddings for each text
        """
        if not self.use_embeddings:
            raise ValueError("Embeddings are not enabled in this LLM configuration.")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise


    # Default sequential query to LLM
    def query_llm(
        self, 
        action: str,
        prompt: str,
        verbose: bool = False,
        **kwargs
    ) -> str:
        """
        Query LLM with action-based prompt dispatch.
        
        Args:
            action: The action to perform (e.g., 'persona_extraction')
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for the API call
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If LLM query fails or action is not supported
        """
        try:
            logger.debug(f"Querying LLM with action '{action}' and prompt length: {len(prompt)}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            if verbose:
                logger.info(f"LLM response for action '{action}': {result}")

            return result

        except Exception as e:
            logger.error(f"LLM query failed for action '{action}': {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create LLM engine
        llm = LLMQueryEngine()
        print(f"LLM engine created successfully. Model: {llm.model}")
        
        # Test with sample action if provided
        if len(sys.argv) > 1:
            action = "persona_extraction"
            test_prompt = " ".join(sys.argv[1:])
            
            result = llm.query_llm(action, test_prompt)
            print(f"\nLLM result for action '{action}':\n{result}")
        else:
            print("\nTo test the LLM engine, provide test utterances as command line arguments.")
            print("Example: python query_llm.py 'Hello there!' 'I really enjoy coding.' 'What do you think?'")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
