
class Filter():
    def get_filter():
        return Filter()

    def __init__(self):
        self.exclude_sentence = []
        self.masked_token = {}

    def add_exclude_sentence(self, sentence):
        if sentence not in self.exclude_sentence:
            self.exclude_sentence.append(sentence)

    def add_mask_token(self, key, value):
        self.masked_token[key] = value

    def filter(self, text):
        if text in self.exclude_sentence:
            return True

        return False

    def replace(self, tokens):
        replaced_tokens = []
        for token in tokens:
            for (key, value) in self.masked_token.items():
                if token == key:
                    token = value
            replaced_tokens.append(token)
        return replaced_tokens
