
import os
import argparse

from flask import Flask, session, request, jsonify, render_template
from datetime import timedelta

from chatbot.interface import ChatBot
from chatbot.filter import Filter
from chatbot.config import config

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(74)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

bot_filter = None
bot = None

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default=1, type=float, required=False, help="生成的temperature")
    parser.add_argument("--repetition_penalty", default=1.2, type=float, required=False, help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument("--topk", default=6, type=int, required=False, help="最高k选1")
    parser.add_argument("--topp", default=0.6, type=float, required=False, help="最高积累概率")
    parser.add_argument("--vocab_path", default=None, type=str, required=False, help="选择词库")
    parser.add_argument("--model_path", default="cambridgeltl/simctg_lccc_dialogue", type=str, required=False, help="对话模型路径")

    return parser.parse_args()

@app.route("/chitchat/history", methods = ["GET"])
def get_history_list():
  global bot
  history_ids = session.get("history")
  if history_ids is None:
      history_ids = []
  history = []
  for history_utr in history_ids:
      history.append( bot.decode(history_utr) )
  return jsonify(history)

@app.route("/chitchat/chat", methods = ["GET"])
def talk():
  global bot_filter
  global bot

  if request.args.get("text"):
    text = request.args.get("text")
    history = session.get("history")
    if history is None:
        history = []
    res, history = bot.chat(history, text, args.repetition_penalty, args.temperature, args.topk, args.topp)
    text = "".join( res )
    session["history"] = history
    return jsonify(text)
  else:
    return jsonify("")

@app.route("/")
def index():
  return "Hello world!"

@app.route("/chitchat", methods = ["GET"])
def chitchat():
    return render_template("chat_template.html")

if __name__ == "__main__":
  args = set_args()
  if bot_filter is None:
      bot_filter = Filter.get_filter()
      for sentence in config["exclude_sentence"]:
          bot_filter.add_exclude_sentence(sentence)
      for key, value in config["mask_token"].items():
          bot_filter.add_mask_token(key, value)

  if bot is None:
      bot = ChatBot.get_chat_bot(args.model_path, None, bot_filter)
  app.run(host="127.0.0.1", port=8080)
