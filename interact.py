
import os

from flask import Flask, session, request, jsonify, render_template
from datetime import timedelta

from transformers import BertTokenizer, GPT2LMHeadModel

import torch
import torch.nn.functional as F

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(74)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

bot = None

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
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

def init_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("cambridgeltl/simctg_lccc_dialogue")
    model = GPT2LMHeadModel.from_pretrained("cambridgeltl/simctg_lccc_dialogue")
    model.to(device)
    model.eval()

    return tokenizer, model


class ChatBot():
  def __init__(self, tokenizer, model):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = tokenizer
    self.model = model

  def ids_to_text(self, ids):
      return "".join( self.tokenizer.convert_ids_to_tokens(ids) )

  def chat(self, history, text, repetition_penalty, temperature):
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
    response = []
    for _ in range(25):
      pt_output = self.model(input_ids)
      logits = pt_output.logits
      next_token_logits = logits[0, -1, :]
      for id in set(response):
        next_token_logits[id] /= repetition_penalty
      #for history_item in history[-5:]:
      #  for word_id in history_item:
      #    next_token_logits[word_id] /= repetition_penalty
      next_token_logits = next_token_logits / temperature
      next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
      filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=8, top_p=1)
      next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
      if next_token == self.tokenizer.sep_token_id:
        break
      input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
      response.append(next_token.item())
    history.append(response)
    return self.tokenizer.convert_ids_to_tokens(response), history

@app.route('/api/history', methods = ['GET'])
def history():
  global bot
  history_ids = session.get("history")
  if history_ids is None:
      history_ids = []
  history = []
  for history_utr in history_ids:
      history.append( bot.ids_to_text(history_utr) )
  return jsonify(history)

@app.route('/api/chat', methods = ['GET'])
def chat():
  global bot
  if request.args.get("text"):
    text = request.args.get("text")
    history = session.get("history")
    if history is None:
        history = []
    req, history = bot.chat(history, text, 1, 1)
    text = "".join( req )
    session["history"] = history
    return jsonify(text)
  else:
    return jsonify("")

@app.route('/', methods = ['GET'])
def index():
  return "Hello world!"

@app.route('/chatroom', methods = ['GET'])
def chatroom():
    return render_template("chatroom.html")

if __name__ == "__main__":
  tokenizer, model = init_model_and_tokenizer()
  bot = ChatBot(tokenizer, model)
  app.run(host="127.0.0.1", port=8080)

