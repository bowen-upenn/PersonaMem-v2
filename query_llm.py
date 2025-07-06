from openai import OpenAI, AzureOpenAI
import timeout_decorator
import utils
import os
import re
from dotenv import load_dotenv
import json


class QueryLLM:
    def __init__(self, args):
        self.args = args
        self.history = []

        load_dotenv()   # Load environment variables
        self._setup_client()


    def _setup_client(self):
        """Setup OpenAI or Azure OpenAI client based on environment variables."""
        # Check for Azure OpenAI configuration first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if azure_endpoint and azure_key and azure_deployment:
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
                    "Microsoft Azure with AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME\n"
                    "or OpenAI with OPENAI_KEY."
                )


    def reset_history(self):
        self.history = []


    @timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # 60s timeout
    def search_images(self, preference):
        """
        Generate image search queries based on user preferences and perform actual web search for images.
        Uses OpenAI's web search capabilities to find relevant images.
        Returns a list of image URLs.
        """
        # Use OpenAI's web search tool to find images
        response = self.client.responses.create(
            model=self.args['models']['llm_model'],
            input=f"Search for images related to this preference: '{preference}'. Find one high-quality image with URL that subtly relate to this preference. Use real-world photos as if they were taken by this user.",
            tools=[{
                "type": "web_search"
            }],
        )

        # Extract image URLs from the response
        print(json.dumps(response.output, default=lambda o: o.__dict__, indent=2))
        response = response.output

        # Parse the response to find image URLs using the https URL pattern
        image_urls = []
        for result in response:
            if 'content' in result:
                text = result['content']['text']
                # Find all URLs in the text using regex
                # This regex captures URLs in markdown format [text](url)
                urls = re.findall(r'\[.*?\]\((https?://[^\s\)]+)\)', text)
                image_urls.extend(urls)

        print(image_urls)
        return image_urls
    

    @timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # Set timeout to 60 seconds
    def query_llm(self, prompt, use_history=False, verbose=False):
        """
        Send a message to the LLM. If use_history is True,
        `prompt` should be a list of message dicts [{'role': ..., 'content': ...}, ...].
        Otherwise, `prompt` is a single string.
        """
        # Prepare messages for the API call
        if use_history:
            self.history.extend([{"role": "user", "content": prompt}])
            messages = self.history
        else:
            messages = [{"role": "user", "content": prompt}]

        # Call the Chat Completions API
        response = self.client.chat.completions.create(
            model=self.args['models']['llm_model'],
            messages=messages,
            max_tokens=self.args['models']['max_tokens'],
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
