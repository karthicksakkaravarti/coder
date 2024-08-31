import torch
import torch.nn as nn
from transformers import BertModel

class HTMLGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_decoder_layers=6):
        super(HTMLGenerator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask, target_ids=None):
        encoder_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        if target_ids is None:
            return self.generate(encoder_output)
        
        tgt_mask = self.generate_square_subsequent_mask(target_ids.size(1)).to(input_ids.device)
        tgt_embeddings = self.bert.embeddings.word_embeddings(target_ids).transpose(0, 1)
        decoder_output = self.decoder(tgt_embeddings, encoder_output.transpose(0, 1), tgt_mask=tgt_mask)
        return self.fc_out(decoder_output.transpose(0, 1))

    def generate(self, encoder_output, max_length=128):
        batch_size = encoder_output.size(0)
        decoder_input = torch.zeros((1, batch_size, self.d_model), device=encoder_output.device)
        
        for _ in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(0)).to(encoder_output.device)
            decoder_output = self.decoder(decoder_input, encoder_output.transpose(0, 1), tgt_mask=tgt_mask)
            next_token_logits = self.fc_out(decoder_output[-1, :, :])
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
            next_token_embed = self.bert.embeddings.word_embeddings(next_token)
            decoder_input = torch.cat([decoder_input, next_token_embed], dim=0)
            
            if next_token.item() == self.bert.config.sep_token_id:
                break
        
        return self.fc_out(decoder_input.transpose(0, 1))

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask