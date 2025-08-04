import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import random

# === Config ===
KG_PATH = "datasets/MultiTQ/kg/full.txt"
INPUT_PATH = "datasets/MultiTQ/negatives2.json"
OUTPUT_PATH = "datasets/MultiTQ/negatives_answer_slot1.json"
QUESTIONS_PATH = "datasets/MultiTQ/questions/train_tear.json"

# === Helpers ===
def standardize_entity(text):
    """엔티티 텍스트 표준화"""
    return str(text).lower().replace('_', ' ').strip()

def get_timestamp_number(date_str):
    try:
        start_date = datetime.strptime("2005-01-01", "%Y-%m-%d").date()
        if isinstance(date_str, str):
            current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        elif isinstance(date_str, datetime):
            current_date = date_str.date()
        else:
            current_date = date_str
        days = (current_date - start_date).days
        return days
    except Exception as e:
        print(f"Error in get_timestamp_number: {e}")
        return None

# === Load resources ===
print("Loading KG triplets...")
with open(KG_PATH) as f:
    triples = [tuple(standardize_entity(x) if i < 3 else x for i, x in enumerate(line.strip().split('\t'))) 
              for line in f if len(line.strip().split('\t')) == 4]

print("Loading questions...")
with open(QUESTIONS_PATH) as f:
    questions_data = json.load(f)
    # question 텍스트를 키로 하는 딕셔너리 생성
    questions = {q['question']: q for q in questions_data}
print(f"Loaded {len(questions)} questions")
print("Sample question:", list(questions.keys())[0])

# === Indexing ===
print("Creating indices...")
sro_index = defaultdict(list)
st_index = defaultdict(list)
ot_index = defaultdict(list)
sot_index = defaultdict(list)

for s, r, o, t in triples:
    sro_index[(s, r, o)].append(t)
    st_index[(s, t)].append((r, o))
    ot_index[(o, t)].append((s, r))
    sot_index[(s, o, t)].append(r)

# === Negative sampling ===
def find_answer_type_based_triples(positive, answer_slot, triples):
    """
    answer type에 따라 네거티브 샘플 후보를 찾는 함수
    - subject: positive의 object 값이 subject 위치에 있는 쿼드러플
    - object: positive의 subject 값이 object 위치에 있는 쿼드러플
    """
    type_based_triples = []
    
    if answer_slot == "subject":
        # positive의 object 값이 subject 위치에 있는 쿼드러플 찾기
        target_value = standardize_entity(positive['object'])
        # tqdm.write(f"\nLooking for object value '{target_value}' in subject position")
        for s, r, o, t in triples:
            if standardize_entity(s) == target_value:
                type_based_triples.append((s, r, o, t))
                
    elif answer_slot == "object":
        # positive의 subject 값이 object 위치에 있는 쿼드러플 찾기
        target_value = standardize_entity(positive['subject'])
        # tqdm.write(f"\nLooking for subject value '{target_value}' in object position")
        for s, r, o, t in triples:
            if standardize_entity(o) == target_value:
                type_based_triples.append((s, r, o, t))
    
    # tqdm.write(f"Found {len(type_based_triples)} type-based candidates")
    return type_based_triples

def generate_negative(positive, triples, question_text):
    """
    네거티브 샘플 생성 함수
    - answer_slot이 subject/object: 5개는 answer type 기반, 5개는 랜덤
    - answer_slot이 timestamp/relation: 10개 모두 랜덤
    """
    negative = []
    
    # question 텍스트로 정보 찾기
    # tqdm.write(f"\nProcessing question: {question_text}")
    question_info = questions.get(question_text)
    # if question_info:
    #     # tqdm.write(f"Found question info: {question_info}")
    # else:
    #     tqdm.write(f"Warning: question not found in questions dictionary")
    
    answer_slot = question_info.get('answer_slot') if question_info else None
    type_based_count = 0
    
    # tqdm.write(f"Question info: {question_info}")
    # tqdm.write(f"Answer slot: {answer_slot}")
    # tqdm.write(f"Positive sample: {positive}")
    
    # 기본 랜덤 샘플링을 위한 후보군 생성 (positive와 같지 않은 것들)
    random_candidates = [t for t in triples if not (
        standardize_entity(t[0]) == standardize_entity(positive['subject']) and 
        standardize_entity(t[1]) == standardize_entity(positive['relation']) and 
        standardize_entity(t[2]) == standardize_entity(positive['object']) and 
        str(get_timestamp_number(t[3])) == str(positive['timestamp'])
    )]
    
    if answer_slot in ["subject", "object"]:
        # answer type 기반 트리플 찾기
        type_based_triples = find_answer_type_based_triples(positive, answer_slot, random_candidates)
        
        # 최대 5개 선택
        if type_based_triples:
            type_samples = random.sample(type_based_triples, min(5, len(type_based_triples)))
            negative.extend(type_samples)
            type_based_count = len(type_samples)
            # tqdm.write(f"Selected {type_based_count} type-based samples")
    
    # 남은 개수를 랜덤 샘플링으로 채우기
    remaining = 10 - len(negative)
    if remaining > 0:
        random_samples = random.sample([t for t in random_candidates if t not in negative], remaining)
        negative.extend(random_samples)
        # tqdm.write(f"Added {remaining} random samples")
    
    return {
        "samples": [{
            "subject": s,
            "relation": r,
            "object": o,
            "timestamp": get_timestamp_number(t)
        } for s, r, o, t in negative],
        "stats": {
            "type_based": type_based_count,
            "random": 10 - type_based_count
        }
    }

# === Main ===
print("Loading existing data...")
with open(INPUT_PATH) as f:
    data = json.load(f)

# 데이터 샘플 체크
print("\nChecking data format...")
print("Total examples:", len(data))
print("Sample example:", data[0])

new_data = []
success_count = 0
fail_count = 0
total_type_based = 0
total_random = 0

# print("\nGenerating negatives...")
for idx, ex in enumerate(tqdm(data)):
    # 기존 positive sample 사용
    positive = {
        "subject": ex["positive"]["subject"],
        "relation": ex["positive"]["relation"],
        "object": ex["positive"]["object"],
        "timestamp": ex["positive"]["timestamp"]
    }
    
    # 새로운 negative 생성 (question 텍스트 사용)
    result = generate_negative(positive, triples, ex.get("question"))
    
    if result["samples"]:
        ex["negative"] = result["samples"]
        new_data.append(ex)
        success_count += 1
        total_type_based += result["stats"]["type_based"]
        total_random += result["stats"]["random"]
    else:
        fail_count += 1
    
    if idx % 1000 == 0:
        avg_type_based = total_type_based / (success_count if success_count > 0 else 1)
        avg_random = total_random / (success_count if success_count > 0 else 1)
        tqdm.write(f"\n[Progress {idx}/{len(data)}]")
        # tqdm.write(f"Success: {success_count}   Fail: {fail_count}")
        # tqdm.write(f"Average negatives per sample:")
        # tqdm.write(f"- Answer type based: {avg_type_based:.2f}")
        # tqdm.write(f"- Random: {avg_random:.2f}")

print("\nSaving results...")
with open(OUTPUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"\n✅ Done: {success_count} succeeded, {fail_count} failed ({(success_count / len(data)) * 100:.2f}%)")
print(f"Final statistics:")
print(f"- Total answer type based negatives: {total_type_based}")
print(f"- Total random negatives: {total_random}")
print(f"- Average answer type based per sample: {total_type_based/success_count:.2f}")
print(f"- Average random per sample: {total_random/success_count:.2f}") 