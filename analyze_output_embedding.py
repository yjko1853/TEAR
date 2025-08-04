import json
import ast
from tqdm import tqdm
import re
import string

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def analyze_output_embedding():
    # 파일 읽기
    # file_name = "top20_predictions_v5_1.json"
    file_name = "test_prompt_timer4_rewrite_penalty_0.8_vote1_temporal_newBERT.json"
    with open(f"datasets/MultiTQ/results/{file_name}", "r") as f:
        data = json.load(f)
    print(file_name)
    total_count = len(data)
    match_count = 0
    
    # 결과를 저장할 리스트
    results = []
    
    for item in tqdm(data, desc="Analyzing"):
        output = item["output"]
        embedding_ids = item["embedding_ids"]
        
        # output이 리스트 형태의 문자열이므로 ast.literal_eval로 변환
        try:
            output_list = ast.literal_eval(output)
        except:
            output_list = [output]
        
        # 정답이 쿼드러플의 어느 위치에 있는지 확인
        found = False
        for quad in embedding_ids:
            # quad는 [s, r, o, t] 형태
            for output_item in output_list:
                output_item = normalize(output_item)
                # s, r, o, t 각각에 대해 정답이 있는지 확인 (부분 문자열 매칭 포함)
                if any(output_item in normalize(element) or normalize(element) in output_item for element in quad):
                    found = True
                    break
            if found:
                break
        
        if found:
            match_count += 1
        
        # 결과 저장
        results.append({
            "output": output,
            "found": found,
            "embedding_ids": embedding_ids
        })
    
    # 결과 저장
    with open("datasets/MultiTQ/results/output_embedding_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 통계 출력
    print(f"\nTotal items: {total_count}")
    print(f"Items with matching output: {match_count}")
    print(f"Match ratio: {(match_count/total_count)*100:.2f}%")

if __name__ == "__main__":
    analyze_output_embedding() 