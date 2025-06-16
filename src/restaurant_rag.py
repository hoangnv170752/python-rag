import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import openai
import json
import logging

from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embeddings_manager import EmbeddingsManager

class RestaurantRAG:
    """
    Specialized RAG system for restaurant and food data queries.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        logging.debug(f"OpenAI API key loaded: {'Present' if self.api_key else 'Missing'}")
        
        self.loader = DocumentLoader('data/documents')
        self.processor = TextProcessor()
        self.embeddings_manager = EmbeddingsManager(self.api_key)
        self.restaurants_data = []
        
        # Initialize system
        logging.info("Initializing RestaurantRAG system")
        self.initialize_system()

    def initialize_system(self):
        """Load restaurant data and prepare for queries"""
        # Load restaurant data from JSON
        logging.info("Loading restaurant data from JSON file")
        self.restaurants_data = self.loader.load_json_data('data_fixed_formatted.json')
        logging.info(f"Loaded {len(self.restaurants_data)} restaurants from data file")
        
        if not self.restaurants_data:
            logging.error("No restaurant data loaded! Check file path and content.")
            logging.debug("Attempting to list available files in the documents directory")
            try:
                import os
                files = os.listdir('data/documents')
                logging.debug(f"Files in data/documents: {files}")
            except Exception as e:
                logging.error(f"Error listing directory: {e}")
        
        # Create searchable text representations for each restaurant and menu item
        logging.info("Creating text representations for restaurants")
        self.restaurant_texts = []
        for i, restaurant in enumerate(self.restaurants_data):
            try:
                # Basic restaurant info
                logging.debug(f"Processing restaurant {i}: {restaurant.get('name', 'Unknown')}")
                restaurant_text = f"Restaurant ID: {restaurant.get('id')}\n"
                restaurant_text += f"Name: {restaurant.get('name')}\n"
                restaurant_text += f"Address: {restaurant.get('address')}\n"
                
                # Menu items
                restaurant_text += "Menu items:\n"
                items = restaurant.get('items', [])
                logging.debug(f"Restaurant has {len(items)} menu items")
                for item in items:
                    restaurant_text += f"- {item.get('name')} - Price: {item.get('price')} VND\n"
                
                self.restaurant_texts.append(restaurant_text)
            except Exception as e:
                logging.error(f"Error processing restaurant {i}: {e}")
                logging.error(f"Restaurant data: {restaurant}")
        
        logging.info(f"Created {len(self.restaurant_texts)} restaurant text representations")
        
        # Create embeddings for restaurant texts
        logging.info("Creating embeddings for restaurant texts")
        self.restaurant_embeddings = self.embeddings_manager.create_embeddings(self.restaurant_texts)
        logging.info(f"Created {len(self.restaurant_embeddings)} restaurant embeddings")

    def search_restaurants(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for restaurants based on a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of matching restaurants with their details
        """
        # Get query embedding
        query_embedding = self.embeddings_manager.create_embeddings([query])[0]
        
        # Find similar restaurants
        similar_restaurants = []
        for i, embedding in enumerate(self.restaurant_embeddings):
            similarity = self.calculate_similarity(query_embedding, embedding)
            similar_restaurants.append((similarity, i))
        
        # Sort by similarity (descending)
        similar_restaurants.sort(reverse=True)
        
        # Return top-k results
        results = []
        for _, idx in similar_restaurants[:top_k]:
            results.append(self.restaurants_data[idx])
        
        return results
    
    def search_menu_items(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for specific menu items across all restaurants
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of matching menu items with restaurant info
        """
        logging.info(f"Searching menu items for query: {query}")
        
        # Get query embedding
        logging.debug("Creating embedding for query")
        query_embedding = self.embeddings_manager.create_embeddings([query])[0]
        
        # Create texts and embeddings for individual menu items
        logging.debug("Creating texts and info for menu items")
        menu_texts = []
        menu_items_info = []
        
        for restaurant in self.restaurants_data:
            try:
                restaurant_id = restaurant.get('id')
                restaurant_name = restaurant.get('name')
                logging.debug(f"Processing menu items for restaurant: {restaurant_name}")
                
                items = restaurant.get('items', [])
                logging.debug(f"Found {len(items)} items in restaurant")
                
                for item in items:
                    try:
                        item_name = item.get('name', 'Unknown')
                        item_price = item.get('price', 0)
                        item_text = f"{item_name} - Price: {item_price} VND"
                        menu_texts.append(item_text)
                        
                        menu_items_info.append({
                            'restaurant_id': restaurant_id,
                            'restaurant_name': restaurant_name,
                            'item': item
                        })
                        logging.debug(f"Added menu item: {item_name}")
                    except Exception as e:
                        logging.error(f"Error processing menu item: {e}")
                        logging.error(f"Item data: {item}")
            except Exception as e:
                logging.error(f"Error processing restaurant menu items: {e}")
                logging.error(f"Restaurant data: {restaurant}")
        
        logging.info(f"Created {len(menu_texts)} menu item texts")
        
        # Create embeddings for menu texts
        logging.debug("Creating embeddings for menu texts")
        menu_embeddings = self.embeddings_manager.create_embeddings(menu_texts)
        logging.debug(f"Created {len(menu_embeddings)} menu embeddings")
        
        # Find similar menu items
        logging.debug("Finding similar menu items")
        similar_items = []
        for i, embedding in enumerate(menu_embeddings):
            similarity = self.calculate_similarity(query_embedding, embedding)
            similar_items.append((similarity, i))
        
        # Sort by similarity (descending)
        similar_items.sort(reverse=True)
        
        # Return top-k results
        results = []
        for _, idx in similar_items[:top_k]:
            results.append(menu_items_info[idx])
        
        logging.info(f"Returning {len(results)} menu item results")
        return results
    
    def calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between two embeddings"""
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def answer_restaurant_query(self, query: str) -> str:
        """
        Answer a query about restaurants or food items
        
        Args:
            query: The user's question about restaurants or food
            
        Returns:
            A natural language response to the query
        """
        # Search for relevant restaurants and menu items
        relevant_restaurants = self.search_restaurants(query)
        relevant_menu_items = self.search_menu_items(query)
        
        # Prepare context
        context = "Restaurant information:\n"
        
        # Add restaurant info
        for restaurant in relevant_restaurants:
            context += f"Restaurant: {restaurant.get('name')}\n"
            context += f"Address: {restaurant.get('address')}\n"
            
            # Add some sample menu items
            context += "Sample menu items:\n"
            for item in restaurant.get('items', [])[:5]:  # Limit to 5 items per restaurant
                context += f"- {item.get('name')} - Price: {item.get('price')} VND\n"
            
            context += "\n"
        
        # Add specific menu items that matched the query
        context += "Specific menu items that match your query:\n"
        for item_info in relevant_menu_items:
            item = item_info['item']
            context += f"- {item.get('name')} - Price: {item.get('price')} VND at {item_info['restaurant_name']}\n"
        
        # Create prompt
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nAnswer:"""
        
        # Get response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in Vietnamese restaurants and food. Use the provided context to answer the question in Vietnamese. If the information is not in the context, say you don't have that information."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
