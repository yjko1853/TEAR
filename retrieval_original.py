import os
import csv
import pickle
import time
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import spacy
import en_core_web_sm
import argparse
import torch
import re
import string
from transrt_model_ver1 import TransRTv2, convert_timestamp_to_days, standardize_entity_format
from tqdm import tqdm

nlp = en_core_web_sm.load()

# 전역 변수로 entity_to_id와 relation_to_id 선언
entity_to_id = {}
relation_to_id = {}

def load_entity_relation_mappings():
    global entity_to_id, relation_to_id
    # 엔티티와 관계 매핑 파일 경로
    entity_path = "datasets/MultiTQ/kg/entity2id.json"
    relation_path = "datasets/MultiTQ/kg/relation2id.json"
    
    # 엔티티 매핑 로드
    with open(entity_path, 'r', encoding='utf-8') as f:
        entity_to_id = json.load(f)
    
    # 관계 매핑 로드
    with open(relation_path, 'r', encoding='utf-8') as f:
        relation_to_id = json.load(f)

def parse_date(date_str):
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %Y"
    ]
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            return date_obj
        except ValueError:
            pass
    return None

def extract_dates(text):
    doc = nlp(text)
    dates = ""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates += ent.text + " "
    dates = dates.strip()
    processed_dates = parse_date(dates)
    return processed_dates

def normalize_text(text):
    """텍스트를 정규화하는 함수 (소문자 변환, 구두점 제거 등)"""
    text = str(text).lower()
    exclude = set(string.punctuation)
    text = "".join(char for char in text if char not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text

def check_answer_in_quadruple(answer: str, quadruple: tuple) -> bool:
    """쿼드러플에서 정답을 찾는 함수"""
    answer = normalize_text(answer)
    for part in quadruple:
        part = normalize_text(part)
        if answer in part or part in answer:   # 양방향 부분 일치 허용
            return True
    return False

class Retrieval:
    def __init__(self, d, model_name, question_list, triple_list, embedding_size=768, transrt_model=None, transrt_weight=0.0, transrt_model_path="saved_models/transrt_model_v1_23.pt", use_subgraph=False, subgraph_path=None):
        self.model = SentenceTransformer(model_name)
        self.embedding_size = embedding_size
        self.question_list = question_list
        self.transrt_model = transrt_model
        self.transrt_weight = transrt_weight
        self.transrt_model_path = transrt_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_subgraph = use_subgraph
        self.subgraph_path = subgraph_path

        if d=='time':
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} from {f[3]} to {f[4]}.' for f in triple_list]
        else:
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} in {f[3]}.' for f in triple_list]

        self.full_time = [triple[3] for triple in triple_list]
        self.triple_list = triple_list
        self.questions = []
        self.question_embedding = None
        self.index = None

        # 서브그래프 사용 시 서브그래프 데이터 로드
        if use_subgraph and subgraph_path:
            print(f"서브그래프 로드 중: {subgraph_path}")
            self.subgraph_data = torch.load(subgraph_path)
            print(f"서브그래프 로드 완료: {len(self.subgraph_data)}개의 질문에 대한 서브그래프")

        # TransRT 모델이 사용되는 경우 매핑 로드
        if transrt_model is not None and transrt_weight > 0:
            load_entity_relation_mappings()

    def build_faiss_index(self):
        return faiss.IndexFlatIP(self.embedding_size)

    def get_embedding(self, text_list, cache_path, is_question=False):
        if os.path.exists(cache_path):
            print(f"캐시된 임베딩 로드 중: {cache_path}")
            return torch.load(cache_path, weights_only=False)
        
        print(f"{'질문' if is_question else '트리플'} 임베딩 계산 중...")
        embeddings = self.model.encode(text_list, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        # 캐시 디렉토리 생성
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True, mode=0o755)
        
        # 임시 파일로 먼저 저장
        temp_path = cache_path + '.tmp'
        try:
            torch.save(embeddings, temp_path)
            # 임시 파일을 최종 파일로 이동
            os.replace(temp_path, cache_path)
            print(f"임베딩 저장 완료: {cache_path}")
        except Exception as e:
            print(f"임베딩 저장 중 오류 발생: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        
        return embeddings

    def compute_similarity(self, n=20):
        """질문과 트리플 간의 유사도를 계산하고 상위 n개를 반환"""
        # 캐시 디렉토리 설정
        cache_dir = "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        def preprocess_text(text):
            """텍스트 전처리 함수"""
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s\.]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # 질문 전처리 및 임베딩 계산
        processed_questions = [preprocess_text(q) for q in self.question_list]
        
        # 질문 임베딩 계산 또는 로드
        question_embeddings_path = os.path.join(cache_dir, "question_embeddings.pt")
        if os.path.exists(question_embeddings_path):
            print(f"캐시된 질문 임베딩 로드 중: {question_embeddings_path}")
            question_embeddings = torch.load(question_embeddings_path, weights_only=False)
        else:
            print("질문 임베딩 계산 중...")
            with torch.no_grad():
                question_embeddings = self.model.encode(
                    processed_questions,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    batch_size=32,
                    normalize_embeddings=True
                )
            torch.save(question_embeddings, question_embeddings_path)
        
        if self.use_subgraph:
            # 서브그래프 처리 코드는 유지
            pass
        else:
            # 전체 KG 사용 시의 코드
            distances = []
            corpus_ids = []
            
            # SentenceBERT 임베딩 계산 (transrt_weight가 1.0이 아닐 때만)
            sbert_index = None
            if self.transrt_weight < 1.0:
                print("\nSentenceBERT 임베딩 준비 중...")
                # 트리플을 텍스트로 변환하고 전처리
                processed_facts = [
                    preprocess_text(f"{s} {r} {o} in {t}.")
                    for s, r, o, t in self.triple_list
                ]
                
                sentbert_embeddings_path = os.path.join(cache_dir, "sentbert_embeddings.pt")
                if os.path.exists(sentbert_embeddings_path):
                    print(f"캐시된 SentenceBERT 임베딩 로드 중: {sentbert_embeddings_path}")
                    sbert_embeddings = torch.load(sentbert_embeddings_path, weights_only=False)
                else:
                    print("SentenceBERT 임베딩 계산 중...")
                    # 배치 크기를 늘려 처리 속도 향상
                    with torch.no_grad():
                        sbert_embeddings = self.model.encode(
                            processed_facts,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            batch_size=64,  # 배치 크기 증가
                            normalize_embeddings=True
                        )
                    torch.save(sbert_embeddings, sentbert_embeddings_path)
                
                # FAISS 인덱스 생성 (SentenceBERT)
                print("SentenceBERT FAISS 인덱스 생성 중...")
                dimension = sbert_embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dimension)
                sbert_index = faiss.IndexIVFFlat(quantizer, dimension, 2048, faiss.METRIC_INNER_PRODUCT)  # 클러스터 수 증가
                sbert_index.nprobe = 256  # 검색 품질 향상을 위해 증가
                
                # 학습 데이터가 충분한지 확인
                if len(sbert_embeddings) < 2048:
                    print("Warning: 데이터가 적어 IVF 클러스터 수를 조정합니다.")
                    n_clusters = max(len(sbert_embeddings) // 39, 1)  # 약 39개의 벡터당 1개의 클러스터
                    sbert_index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
                    sbert_index.nprobe = max(n_clusters // 4, 1)  # nprobe 조정
                
                print("FAISS 인덱스 학습 중...")
                sbert_index.train(sbert_embeddings)
                sbert_index.add(sbert_embeddings)
            
            # TransRT 임베딩 계산 (transrt_weight가 0.0이 아닐 때만)
            transrt_index = None
            if self.transrt_weight > 0.0:
                print("\nTransRT 임베딩 준비 중...")
                transrt_embeddings = self.get_transrt_embeddings()
                
                # FAISS 인덱스 생성 (TransRT)
                print("TransRT FAISS 인덱스 생성 중...")
                dimension = transrt_embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dimension)
                transrt_index = faiss.IndexIVFFlat(quantizer, dimension, 2048, faiss.METRIC_INNER_PRODUCT)
                transrt_index.nprobe = 256
                
                if len(transrt_embeddings) < 2048:
                    n_clusters = max(len(transrt_embeddings) // 39, 1)
                    transrt_index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
                    transrt_index.nprobe = max(n_clusters // 4, 1)
                
                print("TransRT FAISS 인덱스 학습 중...")
                transrt_index.train(transrt_embeddings)
                transrt_index.add(transrt_embeddings)
            
            # 배치 처리로 질문 처리
            print("\n질문 처리 중...")
            batch_size = 32  # 배치 크기 증가
            
            for i in tqdm(range(0, len(question_embeddings), batch_size)):
                batch_questions = question_embeddings[i:i+batch_size]
                batch_distances = []
                batch_indices = []
                
                for q_idx in range(len(batch_questions)):
                    q_emb = batch_questions[q_idx:q_idx+1]
                    combined_scores = []
                    seen_indices = set()
                    
                    # SentenceBERT 검색 (weight가 1.0이 아닐 때)
                    if self.transrt_weight < 1.0:
                        sbert_D, sbert_I = sbert_index.search(q_emb, n * 4)  # 더 많은 후보 검색
                        for s_idx in range(len(sbert_I[0])):
                            idx = sbert_I[0][s_idx]
                            if idx not in seen_indices:
                                seen_indices.add(idx)
                                score = float(sbert_D[0][s_idx]) * (1 - self.transrt_weight)
                                combined_scores.append((score, idx))
                    
                    # TransRT 검색 (weight가 0.0이 아닐 때)
                    if self.transrt_weight > 0.0:
                        transrt_D, transrt_I = transrt_index.search(q_emb, n * 4)
                        for t_idx in range(len(transrt_I[0])):
                            idx = transrt_I[0][t_idx]
                            if idx not in seen_indices:
                                seen_indices.add(idx)
                                score = float(transrt_D[0][t_idx]) * self.transrt_weight
                                combined_scores.append((score, idx))
                            elif self.transrt_weight < 1.0:  # 이미 있는 인덱스에 TransRT 점수 추가
                                for j, (existing_score, existing_idx) in enumerate(combined_scores):
                                    if existing_idx == idx:
                                        total_score = existing_score + float(transrt_D[0][t_idx]) * self.transrt_weight
                                        combined_scores[j] = (total_score, idx)
                                        break
                    
                    # 시간 점수 계산 및 적용
                    question = self.question_list[i + q_idx]
                    target_time = extract_dates(question)
                    if target_time:
                        for j, (score, idx) in enumerate(combined_scores):
                            fact_time = parse_date(self.triple_list[idx][3])
                            if fact_time:
                                time_difference = target_time - fact_time
                                days_difference = time_difference.days
                                time_score = 0.0
                                
                                question_lower = question.lower()
                                if any(word in question_lower for word in ['before', 'prior', 'earlier']):
                                    if 0 < days_difference < 16:
                                        time_score = float(days_difference) / 15.0
                                elif any(word in question_lower for word in ['after', 'later', 'following']):
                                    if -16 < days_difference < 0:
                                        time_score = float(-days_difference) / 15.0
                                elif any(word in question_lower for word in ['in', 'on', 'at', 'during']):
                                    if abs(days_difference) < 1:
                                        time_score = 0.0
                                
                                # 최종 점수 계산 (시맨틱 0.4, 시간 0.6 비율)
                                final_score = score * 0.4 - time_score * 0.6
                                combined_scores[j] = (final_score, idx)
                    
                    # 상위 n개 선택
                    combined_scores.sort(reverse=True)
                    top_n_scores = combined_scores[:n]
                    
                    # 결과 저장
                    batch_distances.append([score for score, _ in top_n_scores])
                    batch_indices.append([idx for _, idx in top_n_scores])
                
                distances.extend(batch_distances)
                corpus_ids.extend(batch_indices)
            
            # 텐서로 변환
            distances = torch.tensor(distances)
            corpus_ids = torch.tensor(corpus_ids)
        
        return distances, corpus_ids

    def get_result(self, distances, corpus_ids, question_list, re_rank=False):
        result_list = []
        for i in range(len(corpus_ids)):
            if re_rank:
                result = self.re_rank_single_result(i, distances[i], corpus_ids[i], question_list)
            else:
                result = self.basic_result(i, distances[i], corpus_ids[i], question_list)
            result_list.append(result)
        return result_list

    def re_rank_single_result(self, i, distances, corpus_ids, question_list, top_k):
        q = question_list[i]
        target_time = extract_dates(q)
        time_list = [10 for _ in range(len(self.full_time))]
        if target_time and target_time != "None":
            self.adjust_time_scores(q, target_time, time_list)
        result = {'question': self.question_list[i]}
        hits = [{'corpus_id': id, 'score': score, 'final_score': score * 0.4 - time_list[id] * 0.6}
                for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)
        result['scores'] = [str(hit['score']) for hit in hits][:top_k]
        result['final_score'] = [str(hit['final_score']) for hit in hits][:top_k]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in hits][:top_k]
        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in hits][:top_k]
        return result

    def adjust_time_scores(self, q, target_time, time_list):
        for idx, t in enumerate(self.full_time):
            t_date = datetime.strptime(t, "%Y-%m-%d").date()
            time_difference = target_time - t_date
            days_difference = time_difference.days
            if 'before' in q:
                if 0 < days_difference < 16:
                    time_list[idx] = days_difference / 15
            elif 'after' in q:
                if -16 < days_difference < 0:
                    time_list[idx] = -days_difference / 15
            elif 'in' in q and days_difference == 0:
                time_list[idx] = 0

    def basic_result(self, i, distances, corpus_ids, question_list):
        result = {'question': question_list[i]}
        
        # 텐서를 CPU로 이동하고 numpy 배열로 변환
        if torch.is_tensor(distances):
            distances = distances.cpu().numpy()
        if torch.is_tensor(corpus_ids):
            corpus_ids = corpus_ids.cpu().numpy()
            
        # 차원 확인 및 처리
        if len(distances.shape) > 1:
            distances = distances.squeeze()
        if len(corpus_ids.shape) > 1:
            corpus_ids = corpus_ids.squeeze()
            
        # 각 요소를 스칼라로 변환
        hits = []
        for id, score in zip(corpus_ids, distances):
            if isinstance(id, (np.ndarray, torch.Tensor)):
                id = id.item() if id.size == 1 else id[0]
            if isinstance(score, (np.ndarray, torch.Tensor)):
                score = score.item() if score.size == 1 else score[0]
            hits.append({'corpus_id': int(id), 'score': float(score)})
            
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        result['scores'] = [str(hit['score']) for hit in hits]
        
        if self.use_subgraph:
            # 서브그래프에서 트리플 가져오기
            subgraph_quadruples = self.subgraph_data[i]['quadruples']
            result['triple'] = [subgraph_quadruples[hit['corpus_id']] for hit in hits]
            result['fact'] = [f'{t[0]} {t[1]} {t[2]} in {t[3]}.' for t in result['triple']]
        else:
            # 전체 KG에서 트리플 가져오기
            result['triple'] = [self.triple_list[hit['corpus_id']] for hit in hits]
            result['fact'] = [self.fact_list[hit['corpus_id']] for hit in hits]
            
        return result

    def save_results(self, result_list, output_path):
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result_list, json_file, indent=4)

    def get_transrt_embeddings(self):
        """TransRT 모델을 사용하여 전체 KG의 쿼드러플에 대한 임베딩을 계산"""
        if self.transrt_model is None:
            raise ValueError("TransRT 모델이 초기화되지 않았습니다.")
        
        # 캐시 파일 경로 설정
        cache_dir = "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        transrt_embeddings_path = os.path.join(cache_dir, "transrt_quadruple_embeddings.pt")
        
        # 캐시된 임베딩이 있는지 확인
        if os.path.exists(transrt_embeddings_path):
            print(f"캐시된 TransRT 임베딩 로드 중: {transrt_embeddings_path}")
            return torch.load(transrt_embeddings_path, weights_only=False)
        
        print(f"\nTransRT 임베딩 계산 시작 (전체 쿼드러플 수: {len(self.triple_list)})")
        
        # 트리플을 TransRT 입력 형식으로 변환
        transrt_inputs = []
        for idx, triple in enumerate(self.triple_list):
            head, relation, tail, time = triple
            # 엔티티와 관계를 ID로 변환
            head_id = entity_to_id.get(standardize_entity_format(head))
            if head_id is None:
                raise ValueError(f"엔티티 ID를 찾을 수 없음 (head): {head}")
            
            relation_id = relation_to_id.get(standardize_entity_format(relation))
            if relation_id is None:
                raise ValueError(f"관계 ID를 찾을 수 없음: {relation}")
            
            tail_id = entity_to_id.get(standardize_entity_format(tail))
            if tail_id is None:
                raise ValueError(f"엔티티 ID를 찾을 수 없음 (tail): {tail}")
            
            # 시간을 일 단위로 변환
            time_days = convert_timestamp_to_days(time)
            transrt_inputs.append((head_id, relation_id, tail_id, time_days))
        
        print(f"전체 {len(transrt_inputs)}개의 쿼드러플에 대한 임베딩 계산 중...")
        
        # TransRT 모델로 임베딩 계산 (배치 처리)
        embeddings = []
        batch_size = 1024
        
        # TransRT 모델을 평가 모드로 설정
        self.transrt_model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(transrt_inputs), batch_size), desc="TransRT 임베딩 계산"):
                batch = transrt_inputs[i:i+batch_size]
                batch_embeddings = []
                
                # 배치 내의 각 쿼드러플에 대해 임베딩 계산
                for head_id, relation_id, tail_id, time_days in batch:
                    embedding = self.transrt_model.get_embedding(head_id, relation_id, tail_id, time_days)
                    batch_embeddings.append(embedding)
                
                # 배치의 임베딩을 스택
                batch_embeddings = torch.stack(batch_embeddings)
                embeddings.append(batch_embeddings)
                
                # 메모리 관리
                torch.cuda.empty_cache()
        
        # 모든 배치의 임베딩을 하나로 합침
        embeddings = torch.cat(embeddings, dim=0)
        print(f"TransRT 임베딩 계산 완료. 임베딩 크기: {embeddings.shape}")
        
        # 크기 확인
        if len(embeddings) != len(self.triple_list):
            raise ValueError(f"계산된 임베딩 수({len(embeddings)})가 전체 쿼드러플 수({len(self.triple_list)})와 일치하지 않습니다.")
        
        # 임베딩 저장
        print(f"TransRT 임베딩 저장 중: {transrt_embeddings_path}")
        torch.save(embeddings.cpu(), transrt_embeddings_path)
        
        return embeddings.cpu().numpy() 