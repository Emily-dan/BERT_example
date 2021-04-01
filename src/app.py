import config
from model import BERTBaseUncased
import torch
import torch.nn as nn

import flask
from flask import Flask
from flask import request

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # unsqueeze(0) 行扩展，一维到二维
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)

    outputs = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs)
    return outputs[0][0].cpu().detach().numpy()


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    positive_pred = sentence_prediction(sentence)
    negative_pred = 1 - positive_pred
    response = {}
    response["response"] = {
        "positive": str(positive_pred),
        "negative": str(negative_pred),
        "sentence": sentence
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()

