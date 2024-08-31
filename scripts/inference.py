import torch
from transformers import BertTokenizer
from src.model import HTMLGenerator
from src.utils import load_config, setup_logging, load_model, get_project_root

import logging
import re
import os


def post_process_html(html):
    # Remove repeated slashes
    html = re.sub(r'/{2,}', '/', html)
    
    # Remove any leading/trailing slashes
    html = html.strip('/')
    
    # Try to close unclosed tags
    open_tags = []
    for tag in re.findall(r'<(\w+)[^>]*>', html):
        if tag not in ['br', 'img', 'input']:  # self-closing tags
            open_tags.append(tag)
    for tag in reversed(open_tags):
        html += f'</{tag}>'
    
    # If still no valid HTML, wrap in paragraph tags
    if not re.search(r'<\w+[^>]*>.*</\w+>', html):
        html = f'<p>{html}</p>'
    
    return html.strip()

def generate_html(model, tokenizer, instruction, device, max_length=128):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(instruction, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        encoder_output = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output_ids = model.generate(encoder_output, max_length=max_length)
        
        generated_html = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return post_process_html(generated_html)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config('configs/model_config.yaml')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Check if the model file exists
        model_path = os.path.join(get_project_root(), config['model_path'])
        if os.path.exists(model_path):
            # Load the trained model
            model = load_model(HTMLGenerator, config['model_path'], device, vocab_size=config['vocab_size'])
            logger.info("Loaded trained model.")
        else:
            # If no trained model exists, initialize a new one
            logger.warning(f"No trained model found at {model_path}. Initializing a new model.")
            model = HTMLGenerator(vocab_size=config['vocab_size'])
            model.to(device)
            model.eval()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Example instructions
        instructions = [
            "Create a heading 11",
            "Create a heading 12",
            "Create a heading 13",
            "Create a heading 14"
        ]

        # Generate HTML for each instruction
        for instruction in instructions:
            generated_html = generate_html(model, tokenizer, instruction, device)
            logger.info(f"Instruction: {instruction}")
            logger.info(f"Generated HTML: {generated_html}\n")

    except Exception as e:
        logger.error(f"An error occurred during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()