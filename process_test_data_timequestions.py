import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fuzzywuzzy import process
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import re

def load_json_file(file_path: str) -> dict:
    print(f"Loading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data) if isinstance(data, list) else 'dict'} from {file_path}")
        return data

def load_id2entity_file(file_path):
    id2entity = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                qid, entity = line.split('\t')
                id2entity[qid] = entity
    return id2entity

def clean_entity(entity: str) -> str:
    # Remove special characters and normalize spaces
    entity = re.sub(r'[^\w\s]', ' ', entity)
    entity = ' '.join(entity.split())
    return entity

def get_main_clause(text: str) -> str:
    # Remove "Before X" or "After X" clauses
    text = re.sub(r'^(Before|After)[^,]+,\s*', '', text)
    return text

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device

def smart_title(s):
    return " ".join(w.capitalize() for w in s.split())

def extract_persons(texts):
    ner_pipe = pipeline(
        "ner",
        model="Jean-Baptiste/roberta-large-ner-english",
        tokenizer="Jean-Baptiste/roberta-large-ner-english",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    out = []
    for sent in tqdm(texts, desc="NER"):
        ents = [(e["word"].lower(), e["score"], "PER")
                for e in ner_pipe(sent)
                if e["entity_group"] == "PER"]
        out.append(ents)
    return out

def extract_countries_from_text(text: str) -> List[str]:
    # 확장된 국가 목록
    common_countries = [
        "Iraq", "China", "Iran", "North Korea", "South Korea", "Japan", "Russia",
        "United States", "India", "Pakistan", "Afghanistan", "Syria", "Israel",
        "Saudi Arabia", "Yemen", "Libya", "Egypt", "Sudan", "Ethiopia", "Somalia",
        "Kenya", "Uganda", "Rwanda", "Congo", "Nigeria", "South Africa", "Ukraine",
        "Turkey", "Germany", "France", "United Kingdom", "Italy", "Spain", "Thailand",
        "Cambodia", "Vietnam", "Malaysia", "Indonesia", "Philippines", "Myanmar",
        "Bangladesh", "Nepal", "Bhutan", "Sri Lanka", "Maldives", "Kazakhstan",
        "Uzbekistan", "Turkmenistan", "Kyrgyzstan", "Tajikistan", "Azerbaijan",
        "Armenia", "Georgia", "Mongolia", "Taiwan", "Laos", "Brunei", "Singapore",
        "East Timor", "Australia", "New Zealand", "Papua New Guinea", "Fiji",
        "Solomon Islands", "Vanuatu", "New Caledonia", "Canada", "Mexico", "Brazil",
        "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador", "Bolivia",
        "Paraguay", "Uruguay", "Guyana", "Suriname", "French Guiana", "Burundi",
        "Tanzania", "Mozambique", "Zimbabwe", "Zambia", "Angola", "Namibia",
        "Botswana", "Lesotho", "Swaziland", "Madagascar", "Mauritius", "Seychelles",
        "Comoros", "Djibouti", "Eritrea"
    ]
    
    found_countries = []
    text_lower = text.lower()
    
    for country in common_countries:
        if country.lower() in text_lower:
            found_countries.append(country)
    
    return found_countries

def find_best_entity_match(extracted, text, kg_name_list, kg_text2id, id2text):
    # ① PER 엔티티 매칭
    best = []
    for ent, prob, lab in extracted:
        if lab.endswith("PER"):
            m = process.extractOne(ent.lower(), kg_name_list, score_cutoff=70)
            if m:
                score = (m[1] * 0.7 + prob * 100 * 0.3) * 1.3
                qid   = kg_text2id[m[0]]
                best.append((qid, score))
    if best:
        best_qid = max(best, key=lambda x: x[1])[0]
        return id2text[best_qid]  # 이름으로 리턴
    # ② 국가 등 (필요하면 여기도 텍스트→id 역-조회)
    # 기존 방식 유지 또는 필요시 추가 구현
    return ""

def main():
    print("Starting processing...")
    start_time = time.time()

    # Set device
    device = get_device()

    # Load data
    test_data = load_json_file('datasets/TimeQuestions/questions/test.json')
    kg_id2text = load_id2entity_file('datasets/TimeQuestions/kg/wd_id2entity_text.txt')
    kg_text2id = {v.lower(): k for k, v in kg_id2text.items()}
    kg_name_list = list(kg_text2id.keys())
    id2text = kg_id2text

    print("Loading NER pipeline...")
    # Get all rewrite questions
    rewrite_questions = [item.get('rewrite_question', item.get('Question', item.get('question', ''))) for item in test_data]
    ner_input_questions = [smart_title(q) for q in rewrite_questions]

    # Pipeline NER 추출
    all_entities = extract_persons(ner_input_questions)

    # 디버깅용: NER 추출 결과 일부 확인
    print("예시 NER 결과:", all_entities[:3])
    print("예시 KG 엔티티:", list(kg_id2text.items())[:3])

    # Process data
    processed_data = []
    print("Processing items and matching entities...")

    for item, rewrite_question, extracted_entities in tqdm(
        zip(test_data, rewrite_questions, all_entities),
        total=len(test_data),
        desc="Processing"
    ):
        topic_entity = find_best_entity_match(extracted_entities, rewrite_question, kg_name_list, kg_text2id, id2text)
        # 대소문자 무시하고 question에 포함되는지 체크
        if topic_entity and topic_entity.lower() in rewrite_question.lower():
            processed_item = dict(item)
            processed_item['topic_entity'] = topic_entity
        else:
            processed_item = dict(item)
            processed_item['topic_entity'] = ""
        processed_data.append(processed_item)

    print("Processing completed. Saving results...")
    # Save processed data
    output_path = 'datasets/TimeQuestions/questions/test_tear_entity.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Results saved to {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == "__main__":
    main() 