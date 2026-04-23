# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch==2.11.0",
#     "transformers==5.6.1",
#     "typer==0.24.1",
#     "accelerate==1.13.0",
# ]
# ///
"""
usage:
uv run --env-file .env --script main.py sample.md
"""

import typer
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

app = typer.Typer()

@app.command()
def check(file: typer.FileText = typer.Argument(..., help="チェック対象のテキストファイル")):
    print("モデルの準備をしています...")
    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter", device_map="auto")

    text = file.read()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("チェック開始します...")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_token_class_ids = outputs.logits.argmax(dim=-1)
    predicted_token_classes = [model.config.id2label[token_id.item()] for token_id in predicted_token_class_ids[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"元の文字列: {text}")
    print("個人情報候補:")
    has_candidate = False
    for token, label in zip(tokens, predicted_token_classes):
        if label != "O":
            has_candidate = True
            print(f"  {token:<15} -> {label}")

    if not has_candidate:
        print("  （候補は見つかりませんでした）")


if __name__ == "__main__":
    app()
