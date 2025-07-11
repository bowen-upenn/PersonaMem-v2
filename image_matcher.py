import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import yaml
import argparse
from tqdm import tqdm

import utils
import prompts
from query_llm import QueryLLM


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
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
    

    def load_or_create_embeddings_cache(self, recreate=False, verbose=False):
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
            self._create_embeddings_database(verbose)
    

    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Task 3: Get sentence embedding for text using SentenceTransformer.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.sentence_model.encode([text])[0]
    

    def _create_embeddings_database(self, verbose=False):
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
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Encode image to base64
            base64_image = utils.encode_image_to_base64(image_path)
            if not base64_image:
                continue
            
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
                image_path = os.path.join(parts[-2], parts[-1])
                demographic_data[image_path] = demographic_info
        
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
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--recreate', dest='recreate', action='store_true', help='Recreate embeddings database even if cache exists')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    print(args)

    # Initialize LLM and ImageMatcher
    llm = QueryLLM(args)
    image_matcher = ImageMatcher(llm)

    # Load or create embeddings cache
    image_matcher.load_or_create_embeddings_cache(recreate=cmd_args.recreate, verbose=args['inference']['verbose'])

    # Check if embeddings database already exists
    stats = image_matcher.get_database_stats()
    print("\n" + "=" * 50)
    print("Database Creation Complete!")
    print(f"✓ Total images processed: {stats['total_images']}")
    print(f"✓ Embedding dimension: {stats['embedding_dimension']}")
    print(f"✓ Cache file: {image_matcher.embeddings_cache_file}")


if __name__ == "__main__":
    main()
