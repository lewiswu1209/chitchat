
from transformers import BertTokenizerFast, GPT2LMHeadModel

import torch
import torch.nn.functional as F

class ChatBot():

  def get_chat_bot(model_path, vocab_path = None, filter=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if vocab_path is None:
      tokenizer = BertTokenizerFast.from_pretrained(model_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    else:
      tokenizer = BertTokenizerFast(vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    special_tokens = []
    for key in filter.masked_token.keys():
      special_tokens.append(key)
    tokenizer.add_special_tokens( {'additional_special_tokens':special_tokens} )
    # tokenizer.add_special_tokens( {'additional_special_tokens':["[NAME]","[NICK]","[GENDER]","[YEAROFBIRTH]","[MONTHOFBIRTH]","[DAYOFBIRTH]","[ZODIAC]","[AGE]","[HEIGHT]","[WEIGHT]"]} )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return ChatBot(tokenizer, model, filter)

  def __init__(self, tokenizer, model, filter):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = tokenizer
    self.model = model
    self.filter = filter

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
    history.append(text_ids)

    input_ids = [self.tokenizer.cls_token_id]
    for history_utr in history[-3:]:
      input_ids.extend(history_utr)
      input_ids.append(self.tokenizer.sep_token_id)

    input_ids = torch.tensor(input_ids).to(self.device)
    input_ids = input_ids.unsqueeze(0)

    while True:
      answer = []
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

      if not self.filter is None:
        if not self.filter.filter(answer):
          break

    history.append(answer)

    if not self.filter is None:
      replaced_answer = self.filter.replace( self.tokenizer.convert_ids_to_tokens(answer) )
    else:
      replaced_answer = self.tokenizer.convert_ids_to_tokens(answer)

    return replaced_answer, history
