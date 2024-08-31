import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class HTMLDataset(Dataset):
    def __init__(self, data_path, max_input_length=64, max_output_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            instruction = self.data.iloc[idx]['instruction']
            html_code = self.data.iloc[idx]['html_code']
            
            instruction_encoding = self.tokenizer(instruction, 
                                                  return_tensors='pt', 
                                                  padding='max_length', 
                                                  truncation=True, 
                                                  max_length=self.max_input_length)
            html_encoding = self.tokenizer(html_code, 
                                           return_tensors='pt', 
                                           padding='max_length', 
                                           truncation=True, 
                                           max_length=self.max_output_length)
            
            return {
                'input_ids': instruction_encoding['input_ids'].squeeze(),
                'attention_mask': instruction_encoding['attention_mask'].squeeze(),
                'labels': html_encoding['input_ids'].squeeze()
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise