from openai import OpenAI, AzureOpenAI
import timeout_decorator
import utils
import os
import re
from dotenv import load_dotenv
import json
import requests
import hashlib
from urllib.parse import urlparse
from typing import Any, List, Dict, Optional
from tqdm.asyncio import tqdm_asyncio, tqdm
import asyncio
import time
import base64


class QueryLLM:
    def __init__(self, args, rate_limit_per_min=50):
        self.args = args
        self.history = []
        self.request_times = []
        self.rate_limit_per_min = rate_limit_per_min
        self.semaphore = asyncio.Semaphore(rate_limit_per_min)  # Max number of concurrent requests

        load_dotenv(override=True)
        self._setup_client()


    def _setup_client(self):
        """Setup OpenAI or Azure OpenAI client based on environment variables."""
        # Check for Azure OpenAI configuration first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_deployment_with_image_input = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_WITH_IMAGE_INPUT")

        print(f"Debug - Environment variables:")
        print(f"  AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
        print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {azure_deployment}")
        print(f"  AZURE_OPENAI_DEPLOYMENT_NAME_WITH_IMAGE_INPUT: {azure_deployment_with_image_input}")
        print(f"  AZURE_OPENAI_API_VERSION: {azure_api_version}")

        if azure_endpoint and azure_key and azure_deployment:
            print("Using Azure OpenAI configuration")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=azure_api_version,
            )
            self.model = azure_deployment
            self.model_with_image_input = azure_deployment_with_image_input
        else:
            try:
                print("Using OpenAI configuration")
                self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
                self.model = self.args['models']['llm_model']
            except Exception as e:
                raise ValueError(
                    "No valid LLM configuration found. Please set either:\n"
                    "Microsoft Azure with AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME\n"
                    "or OpenAI with OPENAI_KEY."
                )


    def reset_history(self):
        self.history = []


    def search_images(self, pref: str):
        """
        Search for images related to a preference using the ImageMatcher.
        This method will be called from conv_generator.py
        
        Args:
            pref: The preference string to search images for
            
        Returns:
            List of image paths related to the preference (top 5 most similar)
        """
        # Import here to avoid circular import
        from image_matcher import ImageMatcher
        
        # Create ImageMatcher instance if not already created
        if not hasattr(self, 'image_matcher'):
            self.image_matcher = ImageMatcher(self.args, self)
        
        # Search for the most similar images (top 5)
        similar_images = self.image_matcher.find_most_similar_image(pref, top_k=5)
        
        if similar_images:
            # Return just the image paths (without similarity scores)
            return [img_path for img_path, _ in similar_images]
        else:
            return []


    # @timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # Set timeout to 60 seconds
    def query_llm(self, prompt, use_history=False, image_path=None, verbose=False):
        # print(f"Querying LLM with prompt: {prompt}\n\n")
        """
        Send a message to the LLM. If use_history is True,
        `prompt` should be a list of message dicts [{'role': ..., 'content': ...}, ...].
        Otherwise, `prompt` is a single string.
        """
        # Prepare messages for the API call
        base64_image = None
        if image_path:
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            except Exception as e:
                print(f"Error reading image file {image_path}: {e}")
        
        if image_path and base64_image:
            curr_message=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ]
            # "type": "image_url",
            # "image_url": {
            #     "url": f"data:image/jpeg;base64,{base64_image}"
            # }
        else:
            curr_message = [{"role": "user", "content": prompt}]

        if use_history:
            self.history.extend(curr_message)
            messages = self.history
        else:
            messages = curr_message

        # Call the Chat Completions API
        response = self.client.chat.completions.create(
            model=self.model_with_image_input if base64_image else self.model,
            messages=messages,
        )

        # Extract content
        try:
            content = response.choices[0].message.content
        except Exception as e:
            print(utils.Colors.WARNING + f'Error getting response: {e}' + utils.Colors.ENDC)
            content = None

        if use_history:
            self.history.append({"role": "assistant", "content": content})

        if verbose:
            print(f'{utils.Colors.OKGREEN}Model Response:{utils.Colors.ENDC} {content}')

        return content


    def _check_rate_limit(self):
        """Check if we can make a request based on rate limit."""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.rate_limit_per_min:
            # Calculate how long to wait
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            return wait_time
        return 0
    

    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limit."""
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            # logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(time.time())


    # @timeout_decorator.timeout(180, timeout_exception=TimeoutError)  # Set timeout to 180 seconds
    async def query_llm_async(self, prompt, use_history=False, image=None, verbose=False):
    
        """
        Async version of query_llm with rate limiting and concurrency control.
        
        Args:
            prompt: The prompt string or list of messages to send to the LLM
            use_history: Whether to use conversation history
            image: Base64 encoded image string (optional)
            verbose: Whether to print debug information
        Returns:
            LLM response text
            
        Raises:
            Exception: If LLM query fails or action is not supported
        """
        async with self.semaphore:  # Limit concurrent requests
            await self._wait_for_rate_limit()  # Respect rate limit
            
            # Prepare messages for the API call
            if image:
                curr_message = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}"
                                }
                            }
                        ]
                    }
                ]
            else:
                curr_message = [{"role": "user", "content": prompt}]

            if use_history:
                self.history.extend(curr_message)
                messages = self.history
            else:
                messages = curr_message

            # Call the Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model_with_image_input if image else self.model,
                messages=messages,
            )

            # Extract content
            try:
                content = response.choices[0].message.content
            except Exception as e:
                print(utils.Colors.WARNING + f'Error getting response: {e}' + utils.Colors.ENDC)
                content = None

            if use_history:
                self.history.append({"role": "assistant", "content": content})

            if verbose:
                print(f'{utils.Colors.OKGREEN}Model Response:{utils.Colors.ENDC} {content}')

            return content


    async def query_llm_parallel(self, prompts, use_history=False, image=None, verbose=False):
        """
        Query LLM in parallel with multiple prompts for the same action while respecting rate limits.
        
        Args:
            action: The action to perform (e.g., 'persona_extraction') - same for all prompts
            prompts: List of prompts to send to the LLM
            verbose: Whether to log verbose output

        Returns:
            List of LLM response texts in the same order as prompts
            
        Example:
            prompts = ["Hello world", "I love coding", "This is a long text..."]
            results = await llm.query_llm_parallel("persona_extraction", prompts)
        """
        async def process_request(prompt):
            return await self.query_llm_async(
                prompt=prompt,
                use_history=use_history,
                image=image,
                verbose=verbose
            )
        
        # Use tqdm for progress tracking
        tasks = [process_request(prompt) for prompt in prompts]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {len(prompts)} LLM requests")
        
        return results
