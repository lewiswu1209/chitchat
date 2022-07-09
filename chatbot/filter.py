
class Filter():

    exclude_sentence = ["图片评论"]

    def filter(text):
        if text in exclude_sentence:
            return True

        return False
