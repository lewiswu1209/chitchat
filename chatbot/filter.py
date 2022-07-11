
from chatbot import config

class Filter():

    def filter(text):
        if text in config.exclude_sentence:
            return True

        return False

    def replace(tokens):
        replaced_tokens = []
        for token in tokens:
            for (key, value) in config.bot_information.items():
                if token == key:
                    token = value
            replaced_tokens.append(token)
        return replaced_tokens