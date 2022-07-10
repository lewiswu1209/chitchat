

from transformers import BertTokenizer, GPT2LMHeadModel

import torch
import torch.nn.functional as F

from chatbot.filter import Filter

class ChatBot():

  def get_chat_bot(pretrained_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    model.to(device)
    model.eval()

    return ChatBot(tokenizer, model)

  def __init__(self, tokenizer, model):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = tokenizer
    self.model = model

  def __top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
      indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
      logits[indices_to_remove] = filter_value
    if top_p > 0.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
      sorted_indices_to_remove = cumulative_probs > top_p
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0
      indices_to_remove = sorted_indices[sorted_indices_to_remove]
      logits[indices_to_remove] = filter_value
    return logits

  def decode(self, ids):
    return "".join( self.tokenizer.convert_ids_to_tokens(ids) )

  def encode(self, text, add_special_tokens=False):
    return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

  def chat(self, history, text, repetition_penalty, temperature, top_k, top_p):
    text_ids = self.tokenizer.encode(text, add_special_tokens=False)

    input_ids = [self.tokenizer.cls_token_id]
    for history_utr in history[-5:]:
      input_ids.extend(history_utr)
      input_ids.append(self.tokenizer.sep_token_id)
    input_ids.extend(text_ids)
    input_ids.append(self.tokenizer.sep_token_id)

    split_text_ids = []
    for text_id in text_ids:
      if text_id != self.tokenizer.sep_token_id:
        split_text_ids.append(text_id)
      else:
        break
    history.append(split_text_ids)

    input_ids = torch.tensor(input_ids).to(self.device)
    input_ids = input_ids.unsqueeze(0)

    answer = []
    while True:
      for _ in range(25):
        output = self.model(input_ids)

        logits = output.logits
        next_token_logits = logits[0, -1, :]
        for id in set(answer):
          next_token_logits[id] /= repetition_penalty
        next_token_logits = next_token_logits / temperature
        next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = self.__top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token == self.tokenizer.sep_token_id:
          break
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        answer.append(next_token.item())

      if not Filter.filter(answer):
        break

    history.append(answer)

    return self.tokenizer.convert_ids_to_tokens(answer), history
