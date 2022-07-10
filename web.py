
import os
import argparse

from flask import Flask, session, request, jsonify, render_template
from datetime import timedelta

from chatbot.interface import ChatBot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(74)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

bot = None

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--vocab_path', default=None, type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='cambridgeltl/simctg_lccc_dialogue', type=str, required=False, help='对话模型路径')

    return parser.parse_args()

@app.route('/api/history', methods = ['GET'])
def get_history_list():
  global bot
  history_ids = session.get("history")
  if history_ids is None:
      history_ids = []
  history = []
  for history_utr in history_ids:
      history.append( bot.decode(history_utr) )
  return jsonify(history)

@app.route('/api/chat', methods = ['GET'])
def chat():
  global bot
  if request.args.get("text"):
    text = request.args.get("text")
    history = session.get("history")
    if history is None:
        history = []
    req, history = bot.chat(history, text, args.repetition_penalty, args.temperature, args.topk, args.topp)
    text = "".join( req )
    session["history"] = history
    return jsonify(text)
  else:
    return jsonify("")

@app.route('/')
def index():
  return "Hello world!"

@app.route('/chatroom', methods = ['GET'])
def chatroom():
    return render_template("chatroom.html")

if __name__ == "__main__":
  args = set_args()

  bot = ChatBot.get_chat_bot(args.model_path, args.vocab_path)
  app.run(host="127.0.0.1", port=8080)
