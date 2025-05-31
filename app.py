from flask import Flask, request, jsonify
import torch
import json
from model import Encoder, Decoder, Seq2Seq  # Save your model classes in model.py
import torch.nn.functional as F

app = Flask(__name__)

# Load vocabs
with open("input_vocab.json", "r", encoding="utf-8") as f:
    input_vocab = json.load(f)
with open("output_vocab.json", "r", encoding="utf-8") as f:
    output_vocab = json.load(f)
inv_output_vocab = {i: c for c, i in output_vocab.items()}

# Initialize model
HIDDEN_SIZE = 512
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)
decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)
model.load_state_dict(torch.load("gru_model.pth", map_location=DEVICE))
model.eval()

# Inference function
def transliterate(text):
    input_ids = [input_vocab["<sos>"]] + [input_vocab.get(c, 0) for c in text] + [input_vocab["<eos>"]]
    src_tensor = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
    src_lens = [len(input_ids)]

    with torch.no_grad():
        encoder_hidden = model.encoder(src_tensor, src_lens)
        decoder_input = torch.tensor([output_vocab["<sos>"]], device=DEVICE)
        decoder_hidden = encoder_hidden

        output_chars = []
        for _ in range(50):  # max length
            output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            top1 = output.argmax(1)
            if top1.item() == output_vocab["<eos>"]:
                break
            output_chars.append(inv_output_vocab.get(top1.item(), ""))
            decoder_input = top1

        return ''.join(output_chars)

@app.route("/transliterate", methods=["POST"])
def transliterate_api():
    data = request.json
    input_text = data.get("text", "")
    output_text = transliterate(input_text)
    return jsonify({"output": output_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

