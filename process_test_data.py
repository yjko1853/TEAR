import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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

def batch_extract_entities(texts: List[str], model, tokenizer, device, batch_size=512) -> List[List[Tuple[str, float]]]:
    all_entities = []
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    
    # Pre-tokenize all texts
    print("Pre-tokenizing all texts...")
    all_inputs = tokenizer(texts, 
                         return_tensors="pt", 
                         truncation=True, 
                         max_length=512, 
                         padding=True,
                         return_attention_mask=True)
    
    try:
        with torch.cuda.amp.autocast():  # Mixed precision
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting entities", total=total_batches):
                batch_end = min(i + batch_size, len(texts))
                
                # Get batch from pre-tokenized inputs
                batch_inputs = {
                    'input_ids': all_inputs['input_ids'][i:batch_end].to(device),
                    'attention_mask': all_inputs['attention_mask'][i:batch_end].to(device)
                }
                
                try:
                    with torch.no_grad():  # Disable gradient calculation
                        outputs = model(**batch_inputs)
                    predictions = torch.argmax(outputs.logits, dim=2)
                    probabilities = torch.softmax(outputs.logits, dim=2)
                    
                    # Move results back to CPU immediately to free GPU memory
                    predictions = predictions.cpu()
                    probabilities = probabilities.cpu()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nGPU out of memory with batch_size={batch_size}. Trying with batch_size={batch_size//2}")
                        torch.cuda.empty_cache()
                        return batch_extract_entities(texts, model, tokenizer, device, batch_size=batch_size//2)
                    else:
                        raise e
                
                # Process batch results
                for tokens, preds, probs in zip(batch_inputs['input_ids'].cpu(), predictions, probabilities):
                    entities = []
                    current_entity = []
                    current_prob = []
                    
                    # Move back to CPU for token conversion
                    token_texts = tokenizer.convert_ids_to_tokens(tokens)
                    
                    for token, pred, prob in zip(token_texts, preds, probs):
                        if pred in [1, 2]:  # B-ENT or I-ENT
                            if token.startswith("##"):
                                if current_entity:
                                    current_entity.append(token[2:])
                                    current_prob.append(float(prob[pred]))
                            else:
                                if current_entity:
                                    entity_text = "".join(current_entity)
                                    entity_prob = sum(current_prob) / len(current_prob)
                                    entities.append((entity_text, entity_prob))
                                    current_entity = []
                                    current_prob = []
                                current_entity.append(token)
                                current_prob.append(float(prob[pred]))
                        else:
                            if current_entity:
                                entity_text = "".join(current_entity)
                                entity_prob = sum(current_prob) / len(current_prob)
                                entities.append((entity_text, entity_prob))
                                current_entity = []
                                current_prob = []
                    
                    if current_entity:
                        entity_text = "".join(current_entity)
                        entity_prob = sum(current_prob) / len(current_prob)
                        entities.append((entity_text, entity_prob))
                    
                    # Filter out special tokens and clean entities
                    cleaned_entities = []
                    for entity, prob in entities:
                        if entity not in ["[CLS]", "[SEP]", "[PAD]"]:
                            cleaned_text = clean_entity(entity)
                            if cleaned_text:  # Only add if not empty after cleaning
                                cleaned_entities.append((cleaned_text, prob))
                    
                    all_entities.append(cleaned_entities)
    except Exception as e:
        print(f"Error in batch_extract_entities: {e}")
        return []
    
    return all_entities

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

def find_best_entity_match(extracted_entities: List[Tuple[str, float]], text: str, kg_entities: Dict[str, int]) -> str:
    # 메인 절에서만 엔티티 추출
    main_text = get_main_clause(text)
    
    # 1. 먼저 텍스트에서 국가명 추출
    countries = extract_countries_from_text(main_text)
    if countries:
        # 국가명이 kg_entities에 있는지 확인
        for country in countries:
            match = process.extractOne(country, kg_entities.keys())
            if match and match[1] > 70:  # 임계값을 70%로 낮춤
                return match[0]
    
    # 2. NER에서 추출된 엔티티 사용
    if not extracted_entities:
        return ""
    
    # 신뢰도와 매칭 점수를 모두 고려
    best_matches = []
    for entity, prob in extracted_entities:
        # 사람 이름이나 국가명 우선 처리
        is_person = any(word.istitle() for word in entity.split())  # 이름은 보통 대문자로 시작
        
        match = process.extractOne(entity, kg_entities.keys())
        if match and match[1] > 70:  # 임계값을 70%로 낮춤
            # 사람 이름이면 가중치 높임
            score_boost = 1.2 if is_person else 1.0
            combined_score = (match[1] * 0.7 + prob * 100 * 0.3) * score_boost
            best_matches.append((match[0], combined_score))
    
    if best_matches:
        return max(best_matches, key=lambda x: x[1])[0]
    return ""

def main():
    print("Starting processing...")
    start_time = time.time()

    # Set device
    device = get_device()

    # Load data
    test_data = load_json_file('datasets/MultiTQ/questions/test_tear_vote.json')
    kg_entities = load_json_file('datasets/MultiTQ/kg/entity2id.json')

    print("Loading NER model...")
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model = model.to(device)  # Move model to GPU
    model.eval()  # Set model to evaluation mode
    print("NER model loaded successfully")

    # Get all rewrite questions
    rewrite_questions = [item['rewrite_question'] for item in test_data]

    # Batch process NER
    print("Extracting entities in batches...")
    with torch.no_grad():  # Disable gradient calculation for inference
        all_entities = batch_extract_entities(rewrite_questions, model, tokenizer, device)

    # Process data
    processed_data = []
    print("Processing items and matching entities...")

    for item, rewrite_question, extracted_entities in tqdm(
        zip(test_data, rewrite_questions, all_entities),
        total=len(test_data),
        desc="Processing"
    ):
        topic_entity = find_best_entity_match(extracted_entities, rewrite_question, kg_entities)
        processed_item = dict(item)
        processed_item['topic_entity'] = topic_entity
        processed_data.append(processed_item)

    print("Processing completed. Saving results...")
    # Save processed data
    output_path = 'datasets/MultiTQ/questions/test_tear_vote+entity.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Results saved to {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == "__main__":
    main() 