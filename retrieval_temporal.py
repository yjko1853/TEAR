import os
import csv
import pickle
import time
import json
import faiss
import numpy as np
from datetime import datetime, date
from sentence_transformers import SentenceTransformer, util
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import argparse
import re
from fine_tuned_retriever_refinetune import SentenceBERTWithTemporalCue, TemporalCueEmbedding
import torch
import torch.nn as nn

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
    processed_dates=parse_date(dates)
    return processed_dates # 移除末尾的空格




class Retrieval:
    def __init__(self, d, model_dir, question_list, triple_list, question_to_slot=None, embedding_size=768, answer_slot_method="masking", topic_entity_list=None, temporal_penalty=0.8):
        # model_dir은 반드시 SentenceTransformer로 저장된 디렉터리 경로여야 함 (pytorch_model.bin 파일 X)
        self.model = SentenceTransformer(model_dir)
        self.embedding_size = embedding_size
        self.question_list = question_list
        self.triple_list = triple_list
        self.question_to_slot = question_to_slot
        self.answer_slot_method = answer_slot_method
        self.topic_entity_list = topic_entity_list
        self.fact_list = []
        self.temporal_penalty = temporal_penalty
        # fact_list 생성
        for f in self.triple_list:
            if len(f) == 4:
                if self.question_to_slot is not None:
                    current_question = self.question_list[len(self.fact_list) % len(self.question_list)]
                    answer_slot = self.question_to_slot.get(current_question, None)
                    if self.answer_slot_method == "masking":
                        if answer_slot == "subject":
                            fact = f"[Answer] {f[1]} {f[2]} in {f[3]}"
                        elif answer_slot == "object":
                            fact = f"{f[0]} {f[1]} [Answer] in {f[3]}"
                        elif answer_slot == "relation":
                            fact = f"{f[0]} [Answer] {f[2]} in {f[3]}"
                        elif answer_slot == "timestamp":
                            fact = f"{f[0]} {f[1]} {f[2]} in [Answer] "
                        else:
                            fact = f"{f[0]} {f[1]} {f[2]} in {f[3]}"
                    else:  # original
                        fact = f"{f[0]} {f[1]} {f[2]} in {f[3]}"
                else:
                    fact = f"{f[0]} {f[1]} {f[2]} in {f[3]}"
                self.fact_list.append(fact)
        self.full_time = [triple[3] for triple in self.triple_list]
        self.triplet_embeddings = None
        self.questions = []
        self.question_embedding = None
        self.index = None

    def build_faiss_index(self, n_clusters=1024, nprobe=128):
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe
        return self.index

    def get_embedding(self, corpus, mode):
        corpus_embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        return corpus_embeddings

    def compute_similarity(self, n):
        self.question_embedding = self.get_embedding(self.question_list, mode="question")
        self.triplet_embeddings = self.get_embedding(self.fact_list, mode="fact")
        index = self.build_faiss_index()
        index.train(self.triplet_embeddings)
        index.add(self.triplet_embeddings)
        print("Performing similarity search...")
        distances, corpus_ids = self.index.search(self.question_embedding, n)
        return distances, corpus_ids


    def extract_temporal_cue(self, question):
        cues = ["first", "last", "before", "after"]
        found_cues = []
        question_lower = question.lower()
        
        for cue in cues:
            if cue in question_lower:
                found_cues.append(cue)
        
        if found_cues:
            print(f"[Temporal Cue 추출] 질문: {question} | Temporal Cues: {', '.join(found_cues)}")
            return found_cues
        else:
            print(f"[Temporal Cue 추출] 질문: {question} | Temporal Cue: None")
            return None

    def extract_target_date(self, question):
        # "before 2020-01-01" 또는 "after 2020-01-01"에서 날짜 추출
        dates = []
        question_lower = question.lower()
        
        # before와 after 모두에 대해 날짜 추출
        for prefix in ['before', 'after']:
            pattern = f'{prefix}\\s+(\\d{{4}}-\\d{{2}}-\\d{{2}})'
            matches = re.finditer(pattern, question_lower)
            for match in matches:
                if match.group(1):  # 날짜 그룹이 있는지 확인
                    dates.append({
                        'type': prefix,
                        'date': match.group(1)
                    })
        
        if dates:
            return dates
        return None

    def group_facts(self, answer_type):
        group_dict = {}
        for idx, triple in enumerate(self.triple_list):
            if answer_type == "Subject":
                key = (triple[1], triple[2])  # (Verb, Object)
            elif answer_type == "Object":
                key = (triple[0], triple[1])  # (Subject, Verb)
            elif answer_type == "Time":
                key = (triple[0], triple[1], triple[2])  # (S, V, O)
            else:
                key = tuple(triple[:3])
            if key not in group_dict:
                group_dict[key] = []
            group_dict[key].append((idx, triple))
        return group_dict

    def group_facts_hits(self, hits, answer_type):
        """
        hits: [{'corpus_id': int, 'score': float}, ...]  ➜ 길이 최대 1000
        answer_type: 'Subject' | 'Object' | 'Time'
        상위 1000개 후보만 이용해 그룹을 만든다.
        """
        group_dict = {}
        for hit in hits:
            idx = hit['corpus_id']
            s, r, o, t = self.triple_list[idx]    # 현재 triple
            if answer_type == "Subject":
                key = (r, o)          # (Verb, Object) 기준
            elif answer_type == "Object":
                key = (s, r)          # (Subject, Verb)
            elif answer_type == "Time":
                key = (s, r, o)       # (S, V, O)
            else:
                key = (s, r, o)       # fallback
            group_dict.setdefault(key, []).append((idx, (s, r, o, t)))
        return group_dict

    def select_representative_fact(self, group, cues, target_dates=None):
        # group: [(idx, triple), ...]
        if not group:
            return None
            
        if target_dates:
            # before/after 날짜가 있는 경우
            valid_facts = []
            for idx, triple in group:
                triple_date = datetime.strptime(triple[3], "%Y-%m-%d")
                is_valid = True
                
                for date_info in target_dates:
                    target_date = datetime.strptime(date_info['date'], "%Y-%m-%d")
                    if date_info['type'] == 'before' and triple_date >= target_date:
                        is_valid = False
                    elif date_info['type'] == 'after' and triple_date <= target_date:
                        is_valid = False
                
                if is_valid:
                    valid_facts.append((idx, triple))
            
            if not valid_facts:
                return None
            group = valid_facts
        
        # first/last 처리
        if 'first' in cues:
            return min(group, key=lambda x: x[1][3])  # 가장 이른 날짜
        elif 'last' in cues:
            return max(group, key=lambda x: x[1][3])  # 가장 늦은 날짜
        
        return None

    def apply_temporal_penalty(self, hits, answer_type, cues, target_dates=None):
        # hits: [{'corpus_id': id, ...}, ...]
        group_dict = self.group_facts_hits(hits, answer_type)
        rep_idx_set = set()
        
        for group in group_dict.values():
            rep = self.select_representative_fact(group, cues, target_dates)
            if rep:
                rep_idx_set.add(rep[0])
        
        for hit in hits:
            if hit['corpus_id'] not in rep_idx_set:
                hit['score'] *= self.temporal_penalty
        return hits

    def get_result(self, distances, corpus_ids, question_list, re_rank=False, answer_types=None):
        result_list = []
        for i in range(len(corpus_ids)):
            result = self.basic_result(i, distances[i], corpus_ids[i], question_list)
            # temporal cue 적용
            cues = self.extract_temporal_cue(question_list[i])
            target_dates = self.extract_target_date(question_list[i])
            answer_type = None
            if answer_types is not None:
                answer_type = answer_types[i]
            if cues and answer_type:
                result_hits = []
                for idx, (cid, score) in enumerate(zip(corpus_ids[i], distances[i])):
                    result_hits.append({'corpus_id': cid, 'score': float(score)})
                result_hits = self.apply_temporal_penalty(result_hits, answer_type, cues, target_dates)
                result_hits = sorted(result_hits, key=lambda x: x['score'], reverse=True)
                result['scores'] = [str(hit['score']) for hit in result_hits]
                result['triple'] = [self.triple_list[hit['corpus_id']] for hit in result_hits]
                result['fact'] =  [self.fact_list[hit['corpus_id']] for hit in result_hits]
            result_list.append(result)
        return result_list

    def re_rank_single_result(self, i, distances, corpus_ids, question_list):
        q = question_list[i] # Timequestions question_list[i]['question']
        target_time = extract_dates(q)
        time_list = [10 for _ in range(len(self.full_time))]
        # print("time_list length : ",len(time_list))
        if target_time and target_time != "None":
            if isinstance(target_time, str):
                target_time = datetime.strptime(target_time, "%Y-%m-%d")
            elif isinstance(target_time, date):
                target_time = datetime.combine(target_time, datetime.min.time()) 
            # target_time = datetime.strptime(target_time, "%Y-%m-%d")
            # print("target_time : ", target_time)
            self.adjust_time_scores(q, target_time, time_list)
        
        filtered_hits = []
        for id, score in zip(corpus_ids, distances):
            triple_time = datetime.strptime(self.full_time[id], "%Y-%m-%d")

            if target_time:
                if "after" in q.lower() and triple_time > target_time:  
                    filtered_hits.append({'corpus_id': id, 'fact': self.fact_list[id], 'score': score})
                elif "before" in q.lower() and triple_time < target_time:  
                    filtered_hits.append({'corpus_id': id, 'fact': self.fact_list[id], 'score': score})
        filtered_hits = [
        {
            'corpus_id': id,
            'fact': self.fact_list[id],
            'score': score,
            'time': time_list[id],  # ✅ `time_list` 반영
            'final_score': score * 0.4 - time_list[id] * 0.6  # ✅ `final_score` 계산 방식 통일
        }
        for id, score in zip(corpus_ids, distances)
        if (("after" in q.lower() and datetime.strptime(self.full_time[id], "%Y-%m-%d") > target_time) or 
            ("before" in q.lower() and datetime.strptime(self.full_time[id], "%Y-%m-%d") < target_time))
        ]
        # 만약 조건을 만족하는 triple이 부족하면 기존 검색 결과에서 채움
        if len(filtered_hits) < 15:
            remaining_hits = [
                {'corpus_id': id, 'fact': self.fact_list[id], 'score': score} for id, score in zip(corpus_ids, distances)
                if id not in [hit['corpus_id'] for hit in filtered_hits]
            ]
            remaining_hits = sorted(remaining_hits, key=lambda x: abs((datetime.strptime(self.full_time[x['corpus_id']], "%Y-%m-%d") - target_time).days))
            filtered_hits.extend(remaining_hits[: (15 - len(filtered_hits))])

        # 최종 15개를 유사도 기준으로 정렬
        
        # for hit in filtered_hits:
        #     hit['final_score'] = hit['score'] * 0.4 - (abs((datetime.strptime(self.full_time[hit['corpus_id']], "%Y-%m-%d") - target_time).days) * 0.6)
        filtered_hits = sorted(filtered_hits, key=lambda x: x['final_score'], reverse=True)[:15]

        result = {'question': self.question_list[i]}
        hits = [{'corpus_id': id,'fact': self.fact_list[id], 'score': score, 'time': time_list[id], 'final_score': score * 0.4 - time_list[id] * 0.6}
                for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)
        print("hits : ", hits[:10])
        result['scores'] = [str(hit['score']) for hit in filtered_hits][:15]
        result['final_score'] = [str(hit['final_score']) for hit in filtered_hits][:15]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in filtered_hits]
        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in filtered_hits]

        return result


    def adjust_time_scores(self, q, target_time, time_list):
        # print("def adjust_time_scores")
        print("q : ",q)
        # print("target_time",target_time)
        for idx, t in enumerate(self.full_time):
            # print("idx , t: ",idx,t)
            t = datetime.strptime(t, "%Y-%m-%d")
            time_difference = target_time - t
            days_difference = time_difference.days
            if 'before' in q or 'Before' in q: # 다른 경우도 있는지 확인할 것
                # print("before")
                if 0 < days_difference < 366:
                    time_list[idx] = days_difference / 365
            elif 'after' in q or 'After' in q:
                # print("after")
                if -366 < days_difference < 0:
                    time_list[idx] = -days_difference / 365
                    # print(time_list[idx])
            elif 'in' in q and days_difference == 0:
                # print("in")
                time_list[idx] = 0
            else:
                print("nothing")
        print("time_list : ", len(time_list))

    def basic_result(self, i, distances, corpus_ids, question_list):
        result = {'question': question_list[i]}
        hits = []
        penalty_count = 0  # 패널티 적용된 fact 개수 카운트
        printed = False  # 한 번만 출력하기 위한 플래그
        for idx, (id, score) in enumerate(zip(corpus_ids, distances)):
            penalty = 1.0
            if self.answer_slot_method == "penalty" and self.topic_entity_list is not None:
                info = self.topic_entity_list[i]
                answer_slot = info.get("answer_slot", None)
                topic_entity = info.get("topic_entity", None)
                
                if answer_slot and topic_entity:
                    topic_entity = topic_entity.lower()
                    fact_str = self.fact_list[id].lower()
                    # 1차: fact 전체에 topic entity가 부분 매칭되는지 확인
                    if topic_entity in fact_str or fact_str in topic_entity:
                        triple = self.triple_list[id]
                        if answer_slot == "subject":
                            slot_value = triple[0].lower()
                        elif answer_slot == "object":
                            slot_value = triple[2].lower()
                        elif answer_slot == "relation":
                            slot_value = triple[1].lower()
                        elif answer_slot == "timestamp":
                            slot_value = triple[3].lower()
                        else:
                            slot_value = ""
                        # 2차: answer slot 위치의 값이 topic entity와 부분 매칭되는지 확인
                        if topic_entity in slot_value or slot_value in topic_entity:
                            penalty = 0.8
                            penalty_count += 1  # 패널티 카운트 증가
                if not printed:
                    print("answer_slot, topic_entity : ", answer_slot, topic_entity, "penalty_count:", penalty_count)
                    printed = True
            hits.append({'corpus_id': id, 'score': float(score) * penalty})
        # for 루프 끝난 뒤에!
        # print("answer_slot, topic_entity : ", answer_slot, topic_entity, "penalty_count:", penalty_count)
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        result['scores'] = [str(hit['score']) for hit in hits]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in hits]
        result['fact'] =  [self.fact_list[hit['corpus_id']] for hit in hits]
        return result

    def save_results(self, result_list, output_path):
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result_list, json_file, indent=4)
