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


def split_into_chunks(text: str, max_chars: int = 800) -> list[str]:
    """テキストを段落ごとに分割し、max_chars 以下のチャンクにまとめる。"""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # 1段落が max_chars を超える場合は強制分割
        while len(para) > max_chars:
            slice_, para = para[:max_chars], para[max_chars:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(slice_)
        if not para:
            continue
        candidate = (current + "\n\n" + para).lstrip() if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


@app.command()
def check(file: typer.FileText = typer.Argument(..., help="チェック対象のテキストファイル")):
    print("モデルの準備をしています...")
    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter", device_map="auto")
    print("model.device:", model.device)

    max_length = model.config.max_position_embeddings
    max_chars = 4096

    text = file.read()
    chunks = split_into_chunks(text, max_chars=max_chars)
    print(f"テキストを {len(chunks)} チャンクに分割しました")

    print("チェック開始します...")
    has_candidate = False
    print("個人情報候補:")
    for chunk_index, chunk in enumerate(chunks, 1):
        print(f"  [チャンク {chunk_index}/{len(chunks)}]")
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        token_ids = inputs["input_ids"][0].tolist()
        predicted_token_classes = [
            model.config.id2label[t.item()]
            for t in outputs.logits[0].argmax(dim=-1)
        ]

        i = 0
        while i < len(token_ids):
            label = predicted_token_classes[i]
            if label == "O":
                i += 1
                continue

            has_candidate = True
            prefix, base_label = label.split("-", 1)
            span_ids = [token_ids[i]]
            i += 1

            if prefix == "B":
                while i < len(token_ids):
                    span_ids.append(token_ids[i])
                    end_label = predicted_token_classes[i]
                    i += 1
                    if end_label.startswith("E-"):
                        break

            decoded = tokenizer.decode(span_ids).strip()
            print(f"  {decoded:<15} -> {base_label}")

    if not has_candidate:
        print("  （候補は見つかりませんでした）")


if __name__ == "__main__":
    app()
