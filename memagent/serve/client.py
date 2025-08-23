# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
import sys
from openai import OpenAI
import logging
import click

class OpenAIClient:
    def __init__(self, port=8000, debug=False):
        # 颜色定义
        self.COLOR = {
            'MODEL': '\033[1;92m',  # 亮绿色
            'SYMBOL': '\033[1;94m', # 亮蓝色 
            'LIST': '\033[1;95m',   # 亮紫色
            'RESET': '\033[0m'      # 重置
        }
        self.session = PromptSession()
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="fake-key"
        )
        
        self.debug = debug
        self._setup_logger()
        
        # 添加配置参数
        self.config = {
            'model': None,
            'temperature': 0.7,
            'top_p': 1.0,
            'max_tokens': None,
            'stream': True,
        }
        self.selected_model = None

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def get_available_models(self):
        try:
            self.logger.debug("正在获取可用模型列表...")
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            if not model_ids:
                p
                return None
            
            # 打印模型列表
            print(f"{self.COLOR['LIST']}可用模型列表:{self.COLOR['RESET']}")
            for i, model in enumerate(model_ids, 1):
                print(f"{self.COLOR['LIST']}  {i}. {model}{self.COLOR['RESET']}")
            
            self.logger.debug(f"获取到 {len(model_ids)} 个可用模型")
            self.selected_model = model_ids[0]
        except Exception as e:
            self.logger.error(f"获取模型失败: {str(e)}", exc_info=self.debug)
            return None

    def _get_model_list(self):
        """获取可用模型列表"""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {str(e)}")
            return None

    def _handle_set_command(self):
        """处理set命令"""
        

        intro = """
<b>配置参数设置</b>
模型(model)\t温度(t)\tTop-p(topp)
是否流式(stream)
返回主界面(q)
请选择设置项或输入q: """
        
        while True:
            choice = prompt(HTML(intro)).strip().lower()
            if choice == 'q':
                return True
            elif choice == 'model':
                status = self._set_model()
            elif choice == 't':
                self._set_temperature()
            elif choice == 'topp':
                self._set_top_p()
            elif choice == 'stream':
                self._set_stream()
            else:
                print("无效选择，请输入model/t/topp或q")

    def run(self):
        self.get_available_models()
        if not self.selected_model:
            print("错误: 没有可用的模型")
            sys.exit(1)

        while True:
            try:
                user_input = self.session.prompt(
                    message=[
                        ('class:model', self.selected_model),
                        ('class:symbol', '$ '),
                    ],
                    style=Style.from_dict({
                        'model': '#90EE90 bold',
                        'symbol': '#87CEFA bold',
                    }),
                )
                
                if user_input.strip().lower() == 'exit':
                    break
                elif user_input.strip().lower() == '$set':
                    self._handle_set_command()
                    continue
                    
                self.get_response(user_input)

                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    def get_response(self, prompt):
        try:
            self.logger.debug(f"发送请求 - 模型: {self.selected_model}")
            completion = self.client.chat.completions.create(
                model=self.selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                stream=self.config['stream'],

            )
            if not self.config['stream']:
                response = completion.choices[0].message.content
                self.logger.debug(f"收到响应")
                print(response)
            else:
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()
        except Exception as e:
            self.logger.error(f"请求失败: {str(e)}", exc_info=self.debug)
            return f"错误: {str(e)}"

    def _set_stream(self):
        """设置stream"""
        while True:
            value = prompt(
                HTML(f"<b>当前stream:</b> {self.config['stream']}\n<b>输入新值(0/1):</b> "),
            )
            if value.lower() in ['0', '1']:
                self.config['stream'] = bool(int(value))
                print(f"stream设置为: {self.config['stream']}")
                return

    def _set_model(self):
        """设置模型"""
        models = self._get_model_list()
        if not models:
            raise ValueError("没有可用的模型")
            
        choices = [f"{i}. {model}" for i, model in enumerate(models, 1)]
        while True:
            selected = prompt(
                HTML(f"<b>选择模型:</b>\n" + "\n".join(choices) + "\n<b>输入序号:</b> "),
            )

            try:
                idx = int(selected) - 1
                self.selected_model = models[idx]
                print(f"模型设置为: {self.selected_model}")
                return
            except (ValueError, IndexError):
                print("无效选择")

    def _set_temperature(self):
        """设置温度"""
        while True:
            try:
                value = prompt(
                    HTML(f"<b>当前温度:</b> {self.config['temperature']}\n<b>输入新值(0-2):</b> "),
                )
                temp = float(value)
                if 0 <= temp <= 2:
                    self.config['temperature'] = temp
                    print(f"温度设置为: {temp}")
                    return
                print("温度必须在0-2之间")
            except ValueError:
                print("请输入有效数字")

    def _set_top_p(self):
        """设置top_p"""
        while True:
            try:
                value = prompt(
                    HTML(f"<b>当前top_p:</b> {self.config['top_p']}\n<b>输入新值(0-1):</b> "),
                )
                top_p = float(value)
                if 0 <= top_p <= 1:
                    self.config['top_p'] = top_p
                    print(f"top_p设置为: {top_p}")
                    return
                print("top_p必须在0-1之间")
            except ValueError:
                print("请输入有效数字")

@click.command()
@click.option('--port', default=8000, help='API端口')
@click.option('--debug/--no-debug', default=False, help='启用调试模式')
def cli(port: str, debug: bool):
    """OpenAI API命令行客户端"""
    client = OpenAIClient(port=port, debug=debug)
    client.run()

if __name__ == "__main__":
    cli()