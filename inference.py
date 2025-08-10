import json
import pandas as pd
import asyncio
import aiohttp
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import yaml
from openai import AzureOpenAI
import argparse
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureOpenAIInference:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the inference engine with Azure OpenAI configuration."""
        # Load environment variables from .env file
        load_dotenv()
        
        self.config = self._load_config(config_path)
        self.client = self._initialize_azure_client()
        self.results_dir = Path("data/model_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add Azure OpenAI specific config for GPT-4.1
        config['azure'] = {
            'api_key': os.getenv('AZURE_OPENAI_KEY'),
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
            'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        }
        
        return config
    
    def _initialize_azure_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client."""
        if not all([self.config['azure']['api_key'], self.config['azure']['endpoint'], 
                   self.config['azure']['deployment_name']]):
            raise ValueError("Missing required Azure OpenAI environment variables. "
                           "Please set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, "
                           "and AZURE_OPENAI_DEPLOYMENT_NAME")
        
        return AzureOpenAI(
            azure_endpoint=self.config['azure']['endpoint'],
            api_key=self.config['azure']['api_key'],
            api_version=self.config['azure']['api_version']
        )
    
    def load_benchmark_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load benchmark data from CSV or JSON file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def create_prompt(self, question: str, persona: str = None, context: str = None) -> str:
        """Create a prompt for the model based on the question and optional context."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        if persona:
            prompt_parts.append(f"Persona: {persona}\n")
        
        prompt_parts.append(f"Question: {question}\n")
        prompt_parts.append("Please provide a detailed and helpful answer to this question.")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(self, question: str, persona: str = None, 
                              context: str = None, model_name: str = None) -> Dict[str, Any]:
        """Generate a response using Azure OpenAI GPT-4.1."""
        try:
            prompt = self.create_prompt(question, persona, context)
            
            response = self.client.chat.completions.create(
                model=self.config['azure']['deployment_name'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides detailed and accurate answers to questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['models']['max_tokens'],
                temperature=0.7
            )
            
            return {
                'question': question,
                'response': response.choices[0].message.content,
                'model': self.config['azure']['deployment_name'],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating response for question: {question[:100]}... Error: {str(e)}")
            return {
                'question': question,
                'response': None,
                'error': str(e),
                'model': self.config['azure']['deployment_name'],
                'timestamp': time.time()
            }
    
    def process_batch(self, batch: List[Dict[str, Any]], model_name: str = None) -> List[Dict[str, Any]]:
        """Process a batch of questions using ThreadPoolExecutor for parallel processing."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(len(batch), 10)) as executor:
            # Create tasks
            future_to_item = {
                executor.submit(
                    asyncio.run, 
                    self.generate_response(
                        item['question'], 
                        item.get('persona'), 
                        item.get('context_file_path'),
                        model_name
                    )
                ): item for item in batch
            }
            
            # Collect results
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item: {str(e)}")
                    results.append({
                        'question': future_to_item[future].get('question', 'Unknown'),
                        'response': None,
                        'error': str(e),
                        'model': self.config['azure']['deployment_name'],
                        'timestamp': time.time()
                    })
        
        return results
    
    def run_inference(self, benchmark_file: str, model_name: str = None, 
                     batch_size: int = 10, max_items: int = None) -> str:
        """Run inference on benchmark data."""
        logger.info(f"Loading benchmark data from {benchmark_file}")
        data = self.load_benchmark_data(benchmark_file)
        
        if max_items:
            data = data[:max_items]
        
        logger.info(f"Processing {len(data)} items with batch size {batch_size}")
        
        # Process data in batches
        all_results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
            
            batch_results = self.process_batch(batch, model_name)
            all_results.extend(batch_results)
            
            # Rate limiting
            if self.config['inference']['rate_limit_per_min'] > 0:
                time.sleep(60 / self.config['inference']['rate_limit_per_min'])
        
        # Save results
        output_file = self.results_dir / f"inference_results_{Path(benchmark_file).stem}_gpt41.json"
        
        results_data = {
            'metadata': {
                'benchmark_file': benchmark_file,
                'model': self.config['azure']['deployment_name'],
                'total_items': len(data),
                'processed_items': len(all_results),
                'batch_size': batch_size,
                'timestamp': time.time()
            },
            'results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Run Azure OpenAI inference on benchmark data')
    parser.add_argument('--benchmark_file', type=str, required=True,
                       help='Path to benchmark CSV or JSON file')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (ignored, always uses GPT-4.1)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for parallel processing')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to process')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = AzureOpenAIInference(args.config)
    
    # Run inference
    output_file = inference.run_inference(
        args.benchmark_file,
        args.model_name,
        args.batch_size,
        args.max_items
    )
    
    print(f"Inference completed. Results saved to: {output_file}")

if __name__ == "__main__":
    main()
