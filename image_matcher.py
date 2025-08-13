import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import argparse
from tqdm import tqdm
import concurrent.futures
import threading
import math

import utils
import prompts
from query_llm import QueryLLM

from dotenv import load_dotenv
from openai import AzureOpenAI


class ImageMatcher:
    def __init__(self, llm):
        """
        Initialize the ImageMatcher with configuration and LLM client.
        
        Args:
            llm: QueryLLM instance for making API calls
        """
        self.llm = llm
        self.image_dir = "data/photobook_images"
        self.embeddings_cache_file = "data/image_embeddings_cache.pkl"

        load_dotenv(override=True)
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBED")

        self.sentence_model = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
        )
        self.sentence_model_name = azure_deployment

        # Database to store image paths and their corresponding embeddings
        self.image_database = {}
        self.embeddings_matrix = None
        self.image_paths = []
    

    def load_images_from_directory(self) -> List[str]:
        """
        Task 1: Load all images from data/photobook_images directory.
        Returns a list of image file paths.
        """
        image_paths = []
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory {self.image_dir} does not exist.")
            return image_paths
        
        # Walk through subdirectories to find JPG images
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
        
        print(f"Found {len(image_paths)} images in {self.image_dir}")
        return image_paths
    

    def load_or_create_embeddings_cache(self, recreate=False, parallel=False, verbose=False):
        """
        Task 4: Load existing embeddings cache or create new one.
        """
        if not recreate and os.path.exists(self.embeddings_cache_file):
            print("Loading existing embeddings cache...")
            with open(self.embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.image_database = cache_data['image_database']
                self.embeddings_matrix = cache_data['embeddings_matrix']
                self.image_paths = cache_data['image_paths']
            print(f"Loaded {len(self.image_paths)} cached embeddings")
        else:
            print("Creating new embeddings cache...")
            self._create_embeddings_database(parallel=parallel, verbose=verbose)
    

    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Task 3: Get sentence embedding for text using SentenceTransformer.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        embedding = self.sentence_model.embeddings.create(
            model=self.sentence_model_name,
            input=[text],
        )
        embedding = embedding.data[0].embedding
        embedding = np.array(embedding)
        return embedding


    def _process_single_image_thread(self, args):
        """
        Thread-safe function to process a single image.
        
        Args:
            args: tuple containing (idx, image_path, verbose)
            
        Returns:
            tuple: (idx, image_path, demographic_info, embedding) or (idx, image_path, None, None) if failed
        """
        idx, image_path, verbose = args
        
        try:
            # Encode image to base64
            base64_image = utils.encode_image_to_base64(image_path)
            if not base64_image:
                return idx, image_path, None, None
            
            # Create message with image and prompt
            prompt = prompts.create_demographic_prompt()
            demographic_info = self.llm.query_llm(prompt, use_history=False, image=base64_image, verbose=verbose)

            # Extract the final output after the last ####
            demographic_info = demographic_info.split("####")[-1].strip()

            if demographic_info:
                # Get embedding for the demographic info
                embedding = self._get_sentence_embedding(demographic_info)
                return idx, image_path, demographic_info, embedding
            else:
                return idx, image_path, None, None
                
        except Exception as e:
            if verbose:
                print(f"Error processing image {idx}: {image_path} - {e}")
            return idx, image_path, None, None


    def _create_embeddings_database(self, parallel=False, verbose=False):
        """
        Task 4: Create embeddings database from images.
        """
        image_paths = self.load_images_from_directory()
        
        if not image_paths:
            print("No images found to create embeddings database.")
            return
        
        embeddings_list = []
        valid_paths = []
        demographic_data = {}  # Dictionary to store demographic info for JSON export
        
        if parallel and hasattr(self.llm, 'rate_limit_per_min'):
            print("Processing images in parallel...")
            
            # Prepare arguments for each image
            image_args = []
            for i, image_path in enumerate(image_paths):
                image_args.append((i, image_path, verbose))
            
            # Process images in parallel batches
            max_workers = min(self.llm.rate_limit_per_min, len(image_paths))
            batch_size = max_workers
            num_batches = math.ceil(len(image_paths) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
                batch_args = image_args[batch_start_idx:batch_end_idx]
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} images)")
                
                # Process batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                    # Submit all tasks for this batch
                    future_to_args = {executor.submit(self._process_single_image_thread, args): args for args in batch_args}
                    
                    # Collect results with progress bar
                    for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                     desc=f"Batch {batch_idx + 1} images", 
                                     total=len(batch_args)):
                        try:
                            idx, image_path, demographic_info, embedding = future.result()
                            
                            if demographic_info is not None and embedding is not None:
                                embeddings_list.append(embedding)
                                valid_paths.append(image_path)
                                
                                # Store in database
                                self.image_database[image_path] = {
                                    'demographic_info': demographic_info,
                                    'embedding': embedding
                                }
                                
                                # Store demographic info for JSON export (using image name as key)
                                parts = image_path.split(os.sep)
                                image_key = os.path.join(parts[-2], parts[-1])
                                demographic_data[image_key] = demographic_info
                                
                                if verbose:
                                    print(f"Successfully processed image {idx}: {image_path}")
                            
                        except Exception as e:
                            args = future_to_args[future]
                            idx = args[0]
                            print(f"Error in future for image {idx}: {e}")
        else:
            print("Processing images sequentially...")
            for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
                if verbose:
                    print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Encode image to base64
                base64_image = utils.encode_image_to_base64(image_path)
                if not base64_image:
                    continue
                
                try:
                    # Create message with image and prompt
                    prompt = prompts.create_demographic_prompt()
                    demographic_info = self.llm.query_llm(prompt, use_history=False, image=base64_image, verbose=verbose)

                    # Extract the final output after the last ####
                    demographic_info = demographic_info.split("####")[-1].strip()

                    if demographic_info:
                        # Get embedding for the demographic info
                        embedding = self._get_sentence_embedding(demographic_info)
                        embeddings_list.append(embedding)
                        valid_paths.append(image_path)
                        
                        # Store in database
                        self.image_database[image_path] = {
                            'demographic_info': demographic_info,
                            'embedding': embedding
                        }
                        
                        # Store demographic info for JSON export (using image name as key)
                        parts = image_path.split(os.sep)
                        image_path_key = os.path.join(parts[-2], parts[-1])
                        demographic_data[image_path_key] = demographic_info
                        
                except Exception as e:
                    if verbose:
                        print(f"Error processing image {i}: {image_path} - {e}")
                    continue
        
        # Create embeddings matrix for fast similarity search
        if embeddings_list:
            self.embeddings_matrix = np.array(embeddings_list)
            self.image_paths = valid_paths
            
            # Save cache
            cache_data = {
                'image_database': self.image_database,
                'embeddings_matrix': self.embeddings_matrix,
                'image_paths': self.image_paths
            }
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save demographic information to JSON file
            demographic_json_file = "data/image_personas.json"
            os.makedirs(os.path.dirname(demographic_json_file), exist_ok=True)
            
            with open(demographic_json_file, 'w', encoding='utf-8') as f:
                json.dump(demographic_data, f, indent=2, ensure_ascii=False)
            print(f"Saved demographic information to {demographic_json_file}")
            print(f"Created embeddings database with {len(valid_paths)} images")
        else:
            print("No valid embeddings were created.")
    

    def find_most_similar_image(self, persona_str: str, top_k: int = 6) -> List[Tuple[str, float]]:
        """
        Args:
            persona_str: The persona string to match against
            top_k: Number of top similar images to return (default: 6)

        Returns:
            List of tuples (image_path, similarity_score) for top k images, or empty list if no images available
        """
        if self.embeddings_matrix is None or len(self.image_paths) == 0:
            print("No embeddings database available. Please create one first.")
            return []
        
        # Get embedding for the persona
        persona_embedding = self._get_sentence_embedding(persona_str)
        persona_embedding = persona_embedding.reshape(1, -1)
        
        # Calculate cosine similarity with all image embeddings
        similarities = cosine_similarity(persona_embedding, self.embeddings_matrix)[0]
        
        # Get top k most similar images
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            image_path = self.image_paths[idx]
            similarity_score = similarities[idx]
            results.append((image_path, similarity_score))
        
        return results
    

    def get_image_demographic_info(self, image_path: str) -> Optional[str]:
        """
        Get stored demographic information for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Demographic information string if available
        """
        if image_path in self.image_database:
            return self.image_database[image_path]['demographic_info']
        return None
    

    def get_database_stats(self) -> Dict:
        """
        Get statistics about the current database.
        
        Returns:
            Dictionary with database statistics
        """
        return {
            'total_images': len(self.image_paths),
            'embedding_dimension': self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            'cache_file_exists': os.path.exists(self.embeddings_cache_file),
            'image_directory_exists': os.path.exists(self.image_dir)
        }


def main():
    """
    Main function to build and save the image embeddings database.
    """
    print("Starting Image Matcher - Building Embeddings Database")
    
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-5-chat", help='Set LLM model. Only applicable for OpenAI. For Microsoft Azure, set the model in .env file.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--recreate', dest='recreate', action='store_true', help='Recreate embeddings database even if cache exists')
    parser.add_argument('--parallel', dest='parallel', action='store_true', help='Process images in parallel using multiple threads')
    parser.add_argument('--rate_limit_per_min', type=int, default=10, help='Rate limit for API calls per minute')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    
    # Set rate limit from command line or config
    rate_limit = cmd_args.rate_limit_per_min if cmd_args.rate_limit_per_min is not None else args['inference'].get('rate_limit_per_min', 10)
    
    print(args)

    # Initialize LLM and ImageMatcher
    llm = QueryLLM(args, rate_limit)
    image_matcher = ImageMatcher(llm)

    # Load or create embeddings cache
    image_matcher.load_or_create_embeddings_cache(recreate=cmd_args.recreate, parallel=cmd_args.parallel, verbose=args['inference']['verbose'])

    # Check if embeddings database already exists
    stats = image_matcher.get_database_stats()
    print("\n" + "=" * 50)
    print("Database Creation Complete!")
    print(f"✓ Total images processed: {stats['total_images']}")
    print(f"✓ Embedding dimension: {stats['embedding_dimension']}")
    print(f"✓ Cache file: {image_matcher.embeddings_cache_file}")


if __name__ == "__main__":
    main()
