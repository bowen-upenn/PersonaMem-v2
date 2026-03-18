from openai import OpenAI, AzureOpenAI
import timeout_decorator
from data_generation import utils
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
import threading
from collections import defaultdict
from dotenv import load_dotenv


# Import Gemini-related libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Gemini models will not be available.")

# Import Claude/Anthropic libraries
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: anthropic not installed. Claude models will not be available.")


class QueryLLM:
    def __init__(self, args, rate_limit_per_min=50):
        self.args = args
        # Use thread-safe dictionary to store history for each thread
        self.thread_histories = defaultdict(list)
        self.request_times = []
        self.rate_limit_per_min = rate_limit_per_min
        self.semaphore = asyncio.Semaphore(rate_limit_per_min)  # Max number of concurrent requests

        # API caching state
        self._current_cache_key = None       # Set by inference.py before each query
        self._gemini_caches = {}             # {cache_key: (CachedContent, GenerativeModel)}

        load_dotenv(override=True)
        self._setup_client()


    def _setup_client(self):
        """Setup OpenAI, Azure OpenAI, Gemini, or Claude client based on environment variables."""
        model_name = self.args['models']['llm_model']
        
        # Check if this is a Gemini model
        if re.search(r'gemini', model_name, re.IGNORECASE):
            if not GEMINI_AVAILABLE:
                raise ValueError("google-generativeai package is not installed. Please install it with: pip install google-generativeai")
            
            gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            
            print(f"Using Google Gemini configuration for model: {model_name}")
            genai.configure(api_key=gemini_api_key)
            self.client = genai
            # Add 'models/' prefix if not already present
            if not model_name.startswith('models/'):
                self.model = f"models/{model_name}"
            else:
                self.model = model_name
            self.is_gemini = True
            self.is_claude = False
            return
        
        # Check if this is a Claude model
        if re.search(r'claude', model_name, re.IGNORECASE):
            if not CLAUDE_AVAILABLE:
                raise ValueError("anthropic package is not installed. Please install it with: pip install anthropic")
            
            claude_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
            if not claude_api_key:
                raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable not set")
            
            print(f"Using Anthropic Claude configuration for model: {model_name}")
            self.client = anthropic.Anthropic(api_key=claude_api_key)
            self.model = model_name
            self.is_gemini = False
            self.is_claude = True
            return
        
        # Original OpenAI/Azure setup
        self.is_gemini = False
        self.is_claude = False
        
        # Check for Azure OpenAI configuration first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        if self.args['models']['llm_model'] == 'gpt-5-chat':
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        else:
            azure_deployment = self.args['models']['llm_model']
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        print(f"Debug - Environment variables:")
        print(f"  AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
        print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {azure_deployment}")
        print(f"  AZURE_OPENAI_API_VERSION: {azure_api_version}")

        if azure_endpoint and azure_key and azure_deployment and azure_api_version:
            print("Using Azure OpenAI configuration")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=azure_api_version,
            )
            self.model = azure_deployment
        else:
            try:
                print("Using OpenAI configuration")
                self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
                self.model = self.args['models']['llm_model']
            except Exception as e:
                raise ValueError(
                    "No valid LLM configuration found. Please set either:\n"
                    "Microsoft Azure with AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, and AZURE_OPENAI_API_VERSION\n"
                    "or OpenAI with OPENAI_KEY\n"
                    "or Google Gemini with GOOGLE_API_KEY or GEMINI_API_KEY\n"
                    "or Anthropic Claude with ANTHROPIC_API_KEY or CLAUDE_API_KEY."
                )


    def reset_history(self, thread_id=None):
        """Reset conversation history for a specific thread or current thread."""
        if thread_id is None:
            thread_id = threading.get_ident()
        
        self.thread_histories[thread_id] = []


    def _openai_to_gemini_history(self, openai_messages):
        """
        Convert OpenAI-style messages to Gemini chat history format.

        Args:
            openai_messages (list of dict): Each dict has "role" and "content".

        Returns:
            list: Gemini-style history (list of dicts with 'role' and 'parts').
        """
        gemini_history = []

        for msg in openai_messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Handle content that might be a list (for multimodal messages)
            if isinstance(content, list):
                # Extract text from multimodal content
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)

            if not content:
                continue  # Skip empty content

            # Map OpenAI roles to Gemini roles
            # Gemini uses 'user' and 'model' (not 'assistant')
            if role == "user" or role == "system":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue  # Skip unsupported roles

            gemini_history.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        return gemini_history


    def _get_or_create_gemini_cache(self, history_messages, cache_key):
        """Create or reuse a Gemini CachedContent for the given history prefix.

        Maintains a dict of caches so 32k and 128k contexts can coexist.
        When a new persona is encountered, old caches for that slot are cleaned up.
        Returns a GenerativeModel from cached content, or None if caching fails.
        """
        import datetime as dt

        # Reuse existing cache if key matches
        if cache_key in self._gemini_caches:
            cached_content, cached_model = self._gemini_caches[cache_key]
            if cached_content is not None:
                return cached_model
            return None  # Previously failed for this key

        # Evict oldest caches if we hit the max size
        max_cache_size = 4
        while len(self._gemini_caches) >= max_cache_size:
            oldest_key = next(iter(self._gemini_caches))
            old_cache, _ = self._gemini_caches.pop(oldest_key)
            if old_cache is not None:
                try:
                    old_cache.delete()
                except Exception:
                    pass

        try:
            from google.generativeai import caching
            cache = caching.CachedContent.create(
                model=self.model,
                contents=history_messages,
                ttl=dt.timedelta(hours=1),
                display_name=f"persona_{cache_key}"
            )
            model = genai.GenerativeModel.from_cached_content(cache)
            self._gemini_caches[cache_key] = (cache, model)
            print(f"  Created Gemini cache for {cache_key}")
            return model
        except Exception as e:
            print(f"  Gemini caching failed (falling back to uncached): {e}")
            self._gemini_caches[cache_key] = (None, None)  # Mark as failed
            return None


    def cleanup_caches(self):
        """Clean up any active API caches."""
        for cache_key, (cached_content, _) in self._gemini_caches.items():
            if cached_content is not None:
                try:
                    cached_content.delete()
                except Exception:
                    pass
        self._gemini_caches.clear()
        print("Cleaned up all Gemini cached content")


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
    def query_llm(self, prompt, use_history=False, image=None, image_path=None, verbose=False, thread_id=None):
        # print(f"Querying LLM with prompt: {prompt}\n\n")
        """
        Send a message to the LLM. If use_history is True,
        `prompt` should be a list of message dicts [{'role': ..., 'content': ...}, ...].
        Otherwise, `prompt` is a single string.
        """
        # Get thread ID for history management
        if thread_id is None:
            thread_id = threading.get_ident()
        
        # Prepare messages for the API call
        if use_history and isinstance(prompt, list):
            # If use_history=True and prompt is already a list of messages, use it directly
            messages = prompt
        else:
            # Handle single prompt case
            if image:
                base64_image = image
            elif image_path:
                try:
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                except Exception as e:
                    base64_image = None
                    print(f"Error reading image file {image_path}: {e}")
            else:
                base64_image = None

            if base64_image:
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
            else:
                curr_message = [{"role": "user", "content": prompt}]

            if use_history:
                self.thread_histories[thread_id].extend(curr_message)
                messages = self.thread_histories[thread_id].copy()
            else:
                messages = curr_message

        # Call the appropriate API based on model type
        if self.is_gemini:
            # Convert OpenAI-style messages to Gemini format
            gemini_messages = self._openai_to_gemini_history(messages)

            try:
                cache_key = self._current_cache_key

                # Attempt cached generation: split into history prefix + final user turn
                if cache_key and len(gemini_messages) > 1:
                    history_prefix = gemini_messages[:-1]
                    final_turn = gemini_messages[-1:]
                    cached_model = self._get_or_create_gemini_cache(history_prefix, cache_key)
                    if cached_model is not None:
                        response = cached_model.generate_content(final_turn)
                    else:
                        # Fallback: send all messages without caching
                        model = self.client.GenerativeModel(self.model)
                        response = model.generate_content(gemini_messages)
                else:
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(gemini_messages)

                content = response.text
            except Exception as e:
                print(utils.Colors.WARNING + f'Error getting Gemini response: {e}' + utils.Colors.ENDC)
                content = None
        elif self.is_claude:
            # Call Claude API with prompt caching
            try:
                # Claude requires separating system messages from the conversation
                # Convert all messages to user/assistant format (Claude doesn't accept 'system' role in messages)
                claude_messages = []
                for msg in messages:
                    role = msg.get('role')
                    content_text = msg.get('content', '')

                    # Handle content that might be a list (for multimodal messages)
                    if isinstance(content_text, list):
                        text_parts = [item.get("text", "") for item in content_text if item.get("type") == "text"]
                        content_text = " ".join(text_parts)

                    # Convert system messages to user messages for Claude
                    if role == 'system':
                        role = 'user'

                    # Only keep user and assistant messages
                    if role in ['user', 'assistant'] and content_text:
                        claude_messages.append({
                            'role': role,
                            'content': content_text
                        })

                # Add cache breakpoint on the chat history prefix (second-to-last message)
                # The last 1-2 messages are the new user query + MCQ instruction;
                # everything before is the stable chat history that should be cached.
                if len(claude_messages) >= 3:
                    cache_idx = len(claude_messages) - 2
                    msg = claude_messages[cache_idx]
                    claude_messages[cache_idx] = {
                        'role': msg['role'],
                        'content': [
                            {
                                'type': 'text',
                                'text': msg['content'],
                                'cache_control': {'type': 'ephemeral'}
                            }
                        ]
                    }

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=claude_messages
                )

                # Log cache usage stats
                usage = response.usage
                cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
                cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                if cache_read > 0 or cache_write > 0:
                    print(f"  Claude cache: read={cache_read}, write={cache_write}, uncached={usage.input_tokens}")

                content = response.content[0].text
            except Exception as e:
                print(utils.Colors.WARNING + f'Error getting Claude response: {e}' + utils.Colors.ENDC)
                content = None
        else:
            # Call OpenAI/Azure OpenAI Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            # Extract content
            try:
                content = response.choices[0].message.content
            except Exception as e:
                print(utils.Colors.WARNING + f'Error getting response: {e}' + utils.Colors.ENDC)
                content = None

        if use_history:
            if isinstance(prompt, list):
                # If prompt was a list of messages, update the entire history
                self.thread_histories[thread_id] = messages + [{"role": "assistant", "content": content}]
            else:
                # If prompt was a single message, just append the assistant response
                self.thread_histories[thread_id].append({"role": "assistant", "content": content})

        if verbose:
            print(f'{utils.Colors.OKGREEN}Model Response:{utils.Colors.ENDC} {content}')

        return content
