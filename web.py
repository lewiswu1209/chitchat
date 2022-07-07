
import os

from flask import Flask, session, request, jsonify, render_template
from datetime import timedelta

from chatbot.interface import ChatBot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(74)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

bot = None

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
    req, history = bot.chat(history, text, 1, 1, 4, 1)
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
  bot = ChatBot.get_chat_bot("cambridgeltl/simctg_lccc_dialogue")
  app.run(host="127.0.0.1", port=8080)
