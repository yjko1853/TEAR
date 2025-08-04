import os

import json

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses, models, util

from torch.utils.data import DataLoader

import torch

import torch.nn as nn

import argparse

import re



TEMPORAL_CUES = ["first", "last", "before", "after"]

CUE2IDX = {cue: i for i, cue in enumerate(TEMPORAL_CUES)}



class TemporalCueEmbedding(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(len(TEMPORAL_CUES), embedding_dim)



    def forward(self, cue):

        # cue: str or None

        if cue is None:

            # 0벡터 반환

            return torch.zeros(self.embedding.embedding_dim, device=self.embedding.weight.device)

        idx = CUE2IDX.get(cue, None)

        if idx is None:

            return torch.zeros(self.embedding.embedding_dim, device=self.embedding.weight.device)

        return self.embedding(torch.tensor(idx, device=self.embedding.weight.device))



class SentenceBERTWithTemporalCue(nn.Module):

    def __init__(self, sbert_model, cue_embedding):

        super().__init__()

        self.sbert = sbert_model

        self.cue_embedding = cue_embedding



    def extract_temporal_cue(self, question):

        for cue in TEMPORAL_CUES:

            if cue in question.lower():

                return cue

        return None



    def forward(self, features):

        # features['texts']: [question, ...]

        question = features['texts'][0]

        cue = self.extract_temporal_cue(question)

        sbert_emb = self.sbert.encode([question], convert_to_tensor=True)[0]

        cue_emb = self.cue_embedding(cue)

        return sbert_emb + cue_emb



def main(args):

    with open(args.path, 'r') as file:

        examples = json.load(file)



    train_size = int(len(examples) * 0.8)

    train_data = []

    for example in examples[:train_size]:

        train_data.append(InputExample(texts=[example['question'], example['positive']], label=1.0))

        for negative in example['negative']:

            train_data.append(InputExample(texts=[example['question'], negative], label=0.0))



    sentences1 = []

    sentences2 = []

    scores = []

    for example in examples[train_size:]:

        # 정답

        sentences1.append(example['question'])

        sentences2.append(example['positive'])

        scores.append(1)

        # 네거티브

        for neg in example['negative']:

            sentences1.append(example['question'])

            sentences2.append(neg)

            scores.append(0)

    assert len(sentences1) == len(sentences2)



    # 기존 체크포인트에서 모델 로드

    word_embedding_model = models.Transformer('/share0/yjko1853/models/retriever/sentenseBERT_multitq', max_seq_length=128)

    tokens = ["[B_SEP]", "[A_SEP]", "[F_SEP]", '[L_SEP]']

    for i in range(4017):

        time_token = f'[Time_{i}]'

        tokens.append(time_token)

    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)

    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))



    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



    # Temporal cue embedding 추가

    cue_embedding = TemporalCueEmbedding(sbert_model.get_sentence_embedding_dimension())

    model = SentenceBERTWithTemporalCue(sbert_model, cue_embedding)



    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    train_dataset = SentencesDataset(train_data, sbert_model)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    train_loss = losses.ContrastiveTensionLossInBatchNegatives(sbert_model, scale=1, similarity_fct=util.dot_score)



    # 모델 학습 (임베딩 합산 구조 반영)

    sbert_model.fit(

        train_objectives=[(train_dataloader, train_loss)],

        epochs=2,

        warmup_steps=1000,

        evaluator=evaluator,

        evaluation_steps=1000,

        optimizer_params={"lr": 5e-5},

        output_path='/share0/bys3158/models/retriever/sentenseBERT_multitq_temporal'

    )



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--path", type=str)
    argparser.add_argument("--model_name", type=str)
    args = argparser.parse_args()

    main(args) 
