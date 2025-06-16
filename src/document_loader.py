import os
import json
import logging
from typing import List, Dict, Union

class DocumentLoader:
    def __init__(self, documents_path: str):
        self.documents_path = documents_path

    def load_documents(self) -> List[str]:
        documents = []
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.documents_path, filename), 'r') as file:
                    documents.append(file.read())
        return documents
    
    def load_json_data(self, filename: str = 'data_fixed_formatted.json') -> List[Dict]:
        """Load and parse JSON data from a file.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            List of dictionaries containing the JSON data
        """
        logging.info(f"Attempting to load JSON data from file: {filename}")
        file_path = os.path.join(self.documents_path, filename)
        logging.debug(f"Full file path: {file_path}")
        
        if os.path.exists(file_path):
            logging.info(f"File exists: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    logging.info(f"Successfully loaded JSON data with {len(data)} items")
                    return data
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                logging.error(f"Error occurred at line {e.lineno}, column {e.colno}")
                # Try to read the problematic line
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        if e.lineno <= len(lines):
                            logging.error(f"Problematic line: {lines[e.lineno-1].strip()}")
                except Exception as read_err:
                    logging.error(f"Could not read problematic line: {read_err}")
                return []
            except Exception as e:
                logging.error(f"Error loading JSON file: {e}")
                return []
        else:
            logging.error(f"File does not exist: {file_path}")
            # List available files in the directory
            try:
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    logging.info(f"Available files in {dir_path}: {files}")
            except Exception as e:
                logging.error(f"Error listing directory: {e}")
            return []