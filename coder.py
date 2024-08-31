import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class HTMLDataset(Dataset):
    def __init__(self, instructions, html_codes, max_input_length=64, max_output_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.instructions = instructions
        self.html_codes = html_codes
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        html_code = self.html_codes[idx]
        
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

class HTMLGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_decoder_layers=6):
        super(HTMLGenerator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask, target_ids=None):
        encoder_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        if target_ids is None:
            return self.generate(encoder_output)
        
        tgt_mask = self.generate_square_subsequent_mask(target_ids.size(1)).to(input_ids.device)
        tgt_embeddings = self.embedding(target_ids).transpose(0, 1)
        decoder_output = self.decoder(tgt_embeddings, encoder_output.transpose(0, 1), tgt_mask=tgt_mask)
        return self.fc_out(decoder_output.transpose(0, 1))

    def generate(self, encoder_output, max_length=128):
        batch_size = encoder_output.size(0)
        decoder_input = torch.zeros((1, batch_size), dtype=torch.long, device=encoder_output.device)
        
        for _ in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(0)).to(encoder_output.device)
            tgt_embeddings = self.embedding(decoder_input)
            decoder_output = self.decoder(tgt_embeddings, encoder_output.transpose(0, 1), tgt_mask=tgt_mask)
            next_token_logits = self.fc_out(decoder_output[-1, :, :])
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
            decoder_input = torch.cat([decoder_input, next_token], dim=0)
            
            if next_token.item() == self.tokenizer.sep_token_id:
                break
        
        return decoder_input.transpose(0, 1)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels[:, :-1])
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Prepare your data
    instructions = ["create a tag with url", "create a heading"]
    html_codes = ["<a href='https://example.com'>Link</a>", "<h1>Heading</h1>"]

    # Create dataset and dataloader
    dataset = HTMLDataset(instructions, html_codes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HTMLGenerator(vocab_size=30522).to(device)  # BERT's vocab size
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'html_generator.pth')

if __name__ == "__main__":
    main()