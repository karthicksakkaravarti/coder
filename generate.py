import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re


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
            
            if next_token.item() == self.bert.config.sep_token_id:
                break
        
        return decoder_input.transpose(0, 1)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def load_model(model_path, device):
    model = HTMLGenerator(vocab_size=30522)  # BERT's vocab size
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def rule_based_html_generation(instruction):
    instruction = instruction.lower()
    if "create a tag with url" in instruction or "create a link" in instruction:
        return '<a href="https://example.com">Link text</a>'
    elif "create a heading" in instruction or "make a header" in instruction:
        return '<h1>Heading</h1>'
    elif "create a list" in instruction:
        return '<ul>\n  <li>Item 1</li>\n  <li>Item 2</li>\n  <li>Item 3</li>\n</ul>'
    else:
        return '<p>Default paragraph for: ' + instruction + '</p>'

def post_process_html(html):
    # Remove repeated slashes
    html = re.sub(r'/{2,}', '/', html)
    
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
        # Tokenize the input instruction
        input_ids = tokenizer.encode(instruction, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate output
        encoder_output = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output_ids = model.generate(encoder_output, max_length=max_length)
        
        # Decode the output
        generated_html = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Post-process the generated HTML
    generated_html = post_process_html(generated_html)
    
    # If the generated HTML is still not valid, use rule-based fallback
    if not re.search(r'<\w+[^>]*>.*</\w+>', generated_html):
        generated_html = rule_based_html_generation(instruction)
    
    return generated_html

def main():
    # Set up device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'html_generator.pth'
    model = load_model(model_path, device)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Example instructions
    instructions = [
       "create a heading"
    ]
    
    # Generate HTML for each instruction
    for instruction in instructions:
        generated_html = generate_html(model, tokenizer, instruction, device)
        print(f"Instruction: {instruction}")
        print(f"Generated HTML: {generated_html}\n")

if __name__ == "__main__":
    main()