import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import HTMLGenerator
from src.dataset import HTMLDataset
from src.utils import load_config, setup_logging, save_model, get_project_root
import logging
import os
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels[:, :-1])
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config('configs/model_config.yaml')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load dataset
        dataset = HTMLDataset(config['data_path'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # Initialize model
        model = HTMLGenerator(vocab_size=config['vocab_size']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

        # Ensure model save directory exists
        model_save_dir = os.path.join(get_project_root(), config['model_save_path'])
        os.makedirs(model_save_dir, exist_ok=True)

        # Training loop
        for epoch in range(config['num_epochs']):
            loss = train(model, dataloader, optimizer, criterion, device, epoch)
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss:.4f}")

            # Print sample prediction
            model.eval()
            with torch.no_grad():
                sample_input = next(iter(dataloader))['input_ids'][:1].to(device)
                sample_output = model.generate(model.bert(sample_input, attention_mask=torch.ones_like(sample_input)).last_hidden_state)
                sample_prediction = dataset.tokenizer.decode(sample_output[0].argmax(dim=-1), skip_special_tokens=True)
                logger.info(f"Sample input: {dataset.tokenizer.decode(sample_input[0], skip_special_tokens=True)}")
                logger.info(f"Sample prediction: {sample_prediction}")

            # Save model checkpoint
            if (epoch + 1) % config['save_every'] == 0:
                save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
                save_model(model, save_path)
                logger.info(f"Model checkpoint saved to {save_path}")

        # Save final model
        final_save_path = os.path.join(get_project_root(), config['model_path'])
        save_model(model, final_save_path)
        logger.info(f"Final model saved to {final_save_path}")

        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()