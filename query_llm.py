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
