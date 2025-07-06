from openai import OpenAI
import timeout_decorator
import utils
import os


class QueryLLM:
    def __init__(self, args):
        self.args = args
        # Load the API key
        with open("api_tokens/openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read().strip()

        self.client = OpenAI(api_key=self.api_key)
        self.history = []


    def reset_history(self):
        self.history = []


    @timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # 60s timeout
    def search_images(self, preference, recency=None, domains=None):
        """
        Use the image_query tool to search for images matching the query.
        Returns a list of image URLs.
        """
        # Define the image_query function schema
        functions = [
            {
                "name": "image_query",
                "description": "Search for images given a text query and provide textual responses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query text"},
                        "recency": {"type": ["integer", "null"], "description": "Filter by recency in days"},
                        "domains": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "Optional list of domains to restrict search"
                        }
                    },
                    "required": ["q"]
                }
            }
        ]

        # Call the ChatCompletion API with a forced tool call
        response = self.client.chat.completions.create(
            model=self.args['models']['llm_model'],
            messages=[{"role": "user", "content": f"Search 3 images that subtly hint at the user’s preferences: {preference[0].lower()}{preference[1:]}. "}],
            functions=functions,
            function_call={"name": "image_query"}
        )

        message = response.choices[0].message
        # Safely check for a function_call attribute
        fc = getattr(message, 'function_call', None)
        if fc and getattr(fc, 'name', None) == "image_query":
            # Parse arguments (if any) and get results
            try:
                # If the tool result is provided in the function_call
                results = getattr(fc, 'result', []) or []
            except Exception:
                results = []
            # Extract URLs
            return [item.get("url") for item in results if isinstance(item, dict) and item.get("url")]

        # No images found or function_call missing
        return []


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
