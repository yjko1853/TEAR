import os
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import DataLoader
import argparse
import numpy as np
import re

# --- 공통 cue encoder 정의 (common_cue_encoder.py로 분리 가능) ---
TEMPORAL_CUES = ["first", "last", "before", "after"]
CUE2IDX = {c: i for i, c in enumerate(TEMPORAL_CUES)}

class SentenceBERTWithTemporalCue(nn.Module):
    def __init__(self, base_ckpt: str, cue_tokens: list):
        super().__init__()
        trans = models.Transformer(base_ckpt, max_seq_length=128)
        trans.tokenizer.add_tokens(cue_tokens, special_tokens=True)
        trans.auto_model.resize_token_embeddings(len(trans.tokenizer))
        pool = models.Pooling(trans.get_word_embedding_dimension())
        self.sbert = SentenceTransformer(modules=[trans, pool])
        dim = self.sbert.get_sentence_embedding_dimension()
        self.cue_embedding = nn.Embedding(len(TEMPORAL_CUES), dim)

    def _extract(self, text: str):
        for c in TEMPORAL_CUES:
            if c in text.lower(): return c
        return None

    def forward(self, texts: list, mode: str):
        base = self.sbert.encode(texts, convert_to_tensor=True)
        if mode == "question":
            cues = [self._extract(t) for t in texts]
            cue = torch.stack([
                self.cue_embedding(torch.tensor(CUE2IDX.get(c, 0), device=base.device)) if c
                else torch.zeros(base.size(-1), device=base.device)
                for c in cues])
        else:  # fact → 0-벡터
            cue = torch.zeros_like(base)
        return base + cue

    def encode_text(self, texts, mode="question"):
        if isinstance(texts, str):
            texts = [texts]
        sb_emb = self.sbert.encode(texts, convert_to_tensor=True)
        if mode == "question":
            cues = [self._extract(t) for t in texts]
            cue = torch.stack([
                self.cue_embedding(torch.tensor(CUE2IDX.get(c, 0), device=sb_emb.device)) if c
                else torch.zeros(sb_emb.size(-1), device=sb_emb.device)
                for c in cues])
        else:  # fact → 0-벡터
            cue = torch.zeros_like(sb_emb)
        return sb_emb + cue

# --- 쿼드러플을 텍스트로 변환 ---
def quad_to_text(s, p, o, t):
    return f"{s} {p} {o} in {t}"

def parse_quadruple(text):
    text = text.strip().rstrip('.')
    m = re.match(r"(.+?) (.+?) (.+?) in ([0-9]{4}-[0-9]{2}-[0-9]{2})$", text)
    if m:
        return [m.group(1), m.group(2), m.group(3), m.group(4)]
    m = re.match(r"(.+?) (.+?) (.+?) ([0-9]{4}-[0-9]{2}-[0-9]{2})$", text)
    if m:
        return [m.group(1), m.group(2), m.group(3), m.group(4)]
    raise ValueError(f"Cannot parse quadruple: {text}")

# --- 논문식 7,8에 맞는 contrastive time loss ---
def contrastive_time_loss(q_emb, f_emb, labels, wp=1.0, wn=1.0):
    sims = torch.nn.functional.cosine_similarity(q_emb, f_emb)
    phi = (sims + 1) / 2
    loss = wp * labels * torch.exp(phi) + wn * (1 - labels) * torch.exp(1 - phi)
    return loss.mean()

# --- PairDataset: (질문, 정답, 네거티브들) → (qs, fs, y) 배치 생성 ---
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.qs = []
        self.fs = []
        self.labels = []
        for ex in data:
            q = ex['question']
            pos = quad_to_text(*parse_quadruple(ex['positive']))
            self.qs.append(q)
            self.fs.append(pos)
            self.labels.append(1.0)
            for neg in ex['negative']:
                neg_quad = quad_to_text(*parse_quadruple(neg))
                self.qs.append(q)
                self.fs.append(neg_quad)
                self.labels.append(0.0)
    def __len__(self):
        return len(self.qs)
    def __getitem__(self, idx):
        return self.qs[idx], self.fs[idx], torch.tensor(self.labels[idx], dtype=torch.float)

# --- 배치 임베딩 함수 ---
def get_embedding(model, corpus, mode):
    embs, bs = [], 64
    with torch.no_grad():
        for i in range(0, len(corpus), bs):
            batch = corpus[i:i+bs]
            embs.append(model.encode_text(batch, mode).cpu())
    emb = torch.cat(embs).numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

# --- 학습 루프 ---
def main(args):
    with open(args.path, 'r') as file:
        examples = json.load(file)

    train_dataset = PairDataset(examples)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)

    cue_tokens = ["[FIRST]", "[LAST]", "[BEFORE]", "[AFTER]"] + [f"[Time_{i}]" for i in range(4017)]
    base_ckpt = "/share0/yjko1853/models/retriever/sentenseBERT_multitq"
    model = SentenceBERTWithTemporalCue(base_ckpt, cue_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        total = 0
        for batch in train_dataloader:
            qs, fs, y = batch
            qs = list(qs)
            fs = list(fs)
            y = y.to(device)
            q_emb = model.encode_text(qs, mode="question")
            f_emb = model.encode_text(fs, mode="fact")
            loss = contrastive_time_loss(q_emb, f_emb, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch{epoch+1}: {total/len(train_dataloader):.4f}")

    # 저장: SentenceTransformer 전체 저장 + cue_embedding/tokens는 extra.pt로 저장
    save_path = args.model_name
    model.sbert.save(save_path)
    torch.save({
        "cue": model.cue_embedding.state_dict(),
        "tokens": cue_tokens
    }, os.path.join(save_path, "extra.pt"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str)
    argparser.add_argument("--model_name", type=str, default="sentenseBERT_multitq_temporal")
    args = argparser.parse_args()
    main(args)