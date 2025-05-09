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
        self.assistant = self.client.beta.assistants.create(
            name="Persona Generator",
            instructions="You are a helpful assistant.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.thread = None


    def create_a_thread(self):
        self.thread = self.client.beta.threads.create()


    @timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # Set timeout to 30 seconds
    def query_llm(self, prompt, use_assistant=False, verbose=False):
        if use_assistant:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=prompt,
            )

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )

            if run.status == 'completed':
                response = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id,
                )
                try:
                    response = response.data[0].content[0].text.value
                except Exception as e:
                    print(utils.Colors.WARNING + f'Error getting response: {e}' + utils.Colors.ENDC)
                    response = None

                if verbose:
                    print(f'{utils.Colors.OKGREEN}Model Response:{utils.Colors.ENDC} {response}')
            else:
                response = None
                print(run.status)

        else:
            response = self.client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=prompt,
                max_tokens=self.args['llm']['max_tokens'],
            )
            try:
                response = response.choices[0].message.content
            except Exception as e:
                print(utils.Colors.WARNING + f'Error getting response: {e}' + utils.Colors.ENDC)
                response = None

        return response
