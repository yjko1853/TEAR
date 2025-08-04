import os
import json
import openai
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
import asyncio
from typing import List, Dict, Tuple, Set
import glob
import time
from datetime import datetime

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API Key가 없습니다!")

# 기본 설정
DEFAULT_TEMPERATURE = 0.5
DEFAULT_N_SAMPLES = 3
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_TOKENS = 1
DEFAULT_CONCURRENT_CALLS = 2
DEFAULT_RATE_LIMIT_DELAY = 0.2
MAX_RETRIES = 3
VALID_ANSWER_SLOTS: Set[str] = {"subject", "object", "relation", "timestamp"}
CHECKPOINT_INTERVAL = 100  # 100개 질문마다 체크포인트 저장

BATCH_PROMPT_TEMPLATE = """You are an answer type classifier.
You will be given multiple questions, and for each question you must decide which of the following answer types the question requires:
subject, relation, object, timestamp

Example:
Q: Which country was the last to accuse the UN Security Council before the Military Personnel of Canada did?
A: subject
Q: Who was the first to appeal for the indigenous people of Kenya, after the East African Community?
A: subject
Q: To whom did Nicos Anastasiades make his last appeal before the Greek Prime Minister?
A: object
Q: With whom did Federica Mogherini announce her intention to negotiate after 7 May 2015?
A: object
Q: When did the al-Shabaab insurgency use unconventional violence against Muslims in the United Kingdom?
A: timestamp
Q: In which month did the militant of Taliban praise the government of Pakistan for the first time?
A: timestamp

Questions:
{questions}

Please provide answers in the following format:
Q1: [answer]
Q2: [answer]
...
"""

def build_batch_prompt(questions: List[str]) -> str:
    """배치 프롬프트 템플릿에 질문들을 적용"""
    numbered_questions = [f"Q{i+1}: {q}" for i, q in enumerate(questions)]
    return BATCH_PROMPT_TEMPLATE.format(questions="\n".join(numbered_questions))

def parse_batch_response(response: str, n_questions: int) -> List[str]:
    """배치 응답을 파싱하여 각 질문의 답변을 추출"""
    answers = []
    for i in range(n_questions):
        try:
            # Q{i+1}: [answer] 형식에서 answer 추출
            answer = response.split(f"Q{i+1}:")[1].split("\n")[0].strip().lower()
            if answer in VALID_ANSWER_SLOTS:
                answers.append(answer)
            else:
                answers.append("")  # 유효하지 않은 답변
        except:
            answers.append("")  # 파싱 실패
    return answers

async def classify_batch_questions(questions: List[str], client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                                 n_samples: int = DEFAULT_N_SAMPLES,
                                 temperature: float = DEFAULT_TEMPERATURE) -> Tuple[List[str], Dict[str, int]]:
    """
    배치 단위로 질문들을 처리하고 voting으로 결과 결정
    """
    all_answers = [[] for _ in range(len(questions))]
    retry_count = 0
    
    async with semaphore:
        while retry_count < MAX_RETRIES:
            try:
                prompt = build_batch_prompt(questions)
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=len(questions) * 10,  # 각 질문당 충분한 토큰 할당
                    n=n_samples  # 한 번의 호출로 n_samples만큼 응답 받기
                )
                
                # # 응답 개수 확인
                # print(f"\n=== 배치 응답 정보 ===")
                # print(f"요청한 샘플 수: {n_samples}")
                # print(f"실제 응답 개수: {len(response.choices)}")
                
                # # 각 응답 내용 출력
                # for idx, choice in enumerate(response.choices):
                #     print(f"\n== 응답 {idx+1} ==")
                #     print(f"{choice.message.content}\n")
                
                # 각 응답에 대해 처리
                for choice in response.choices:
                    answers = parse_batch_response(choice.message.content, len(questions))
                    for i, ans in enumerate(answers):
                        if ans:  # 유효한 답변만 추가
                            all_answers[i].append(ans)
                break
                    
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"Rate limit exceeded after {MAX_RETRIES} retries. Skipping batch...")
                    break
                wait_time = int(e.response.headers.get("Retry-After", 0)) or DEFAULT_RATE_LIMIT_DELAY * 2**(retry_count-1)
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                print(f"에러 발생: {str(e)}")
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    break
                continue
    
    # 각 질문별로 최종 결과 결정
    final_answers = []
    vote_stats = {"first": 0, "second": 0, "third": 0, "fourth": 0}
    
    for answers in all_answers:
        if not answers:
            final_answers.append("")
            continue
            
        counter = Counter(answers)
        most_common = counter.most_common()
        
        # 동점 처리
        if len(most_common) > 1 and most_common[0][1] == most_common[-1][1]:
            final_answers.append(answers[0])
            vote_stats["fourth"] += 1
        else:
            final_answers.append(most_common[0][0])
            vote_stats["first"] += 1
    
    return final_answers, vote_stats

def save_checkpoint(results: List[Dict], start_idx: int, output_path: str):
    """체크포인트 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{output_path}.{start_idx}_{timestamp}.jsonl"
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"체크포인트 저장 완료: {checkpoint_path}")
    return checkpoint_path

def load_latest_checkpoint(output_path: str) -> Tuple[List[Dict], int]:
    """가장 최근 체크포인트 로드"""
    checkpoint_files = glob.glob(f"{output_path}.*.jsonl")
    if not checkpoint_files:
        return [], 0
    
    latest_file = max(checkpoint_files, key=os.path.getctime)
    results = []
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    # 마지막 처리된 인덱스 추출
    last_idx = int(latest_file.split('.')[-2].split('_')[0])
    
    return results, last_idx

async def process_questions_batch(questions: List[str], client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                                output_path: str = None) -> List[Dict]:
    """배치 단위로 질문 처리 및 결과 저장"""
    all_results = []
    start_idx = 0
    
    # 이전 체크포인트 로드
    if output_path:
        all_results, start_idx = load_latest_checkpoint(output_path)
        print(f"이전 체크포인트 로드 완료: {len(all_results)}개, 마지막 인덱스: {start_idx}")
    
    try:
        # 배치 단위로 처리
        for i in range(start_idx, len(questions), DEFAULT_BATCH_SIZE):
            batch_end = min(i + DEFAULT_BATCH_SIZE, len(questions))
            print(f"배치 처리 중: {i} to {batch_end}")
            
            batch = questions[i:batch_end]
            batch_results, vote_stats = await classify_batch_questions(batch, client, semaphore)
            
            # 결과 형식화
            for j, (question, answer_slot) in enumerate(zip(batch, batch_results)):
                result = {
                    "quid": question['quid'],
                    "question": question['question'],
                    "rewrite_question": question['question'],  # 나중에 merge_answer_slot_results.py에서 실제 rewrite_question으로 교체됨
                    "answers": question['answers'],
                    "answer_slot": answer_slot,
                    "answer_type": question['answer_type'],
                    "time_level": question['time_level'],
                    "qtype": question['qtype'],
                    "qlabel": question['qlabel']
                }
                all_results.append(result)
            
            # 체크포인트 저장
            if len(all_results) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(all_results, i, output_path)
                print(f"현재 투표 통계: {vote_stats}")
    
    except Exception as e:
        print(f"처리 중 에러 발생: {str(e)}")
        if output_path and all_results:
            save_checkpoint(all_results, i, output_path)
    
    return all_results

async def main():
    # train.json 파일 읽기
    input_file = "/home/bys3158/timequestions/datasets/MultiTQ/questions/test.json"
    output_file = "/home/bys3158/timequestions/datasets/MultiTQ/results/answer_slot_test_vote_1call_6samples_0.5.json"
    
    # 입력 파일 읽기
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"입력 파일을 읽는 중 오류 발생: {str(e)}")
        exit(1)
    
    # 세마포어와 클라이언트 생성
    semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_CALLS)
    client = AsyncOpenAI(
        api_key=api_key,
        max_retries=5,
        timeout=30
    )
    
    # 파이프라인 실행
    print(f"총 {len(data)}개의 질문을 처리합니다...")
    results = await process_questions_batch(
        questions=data,
        client=client,
        semaphore=semaphore,
        output_path=output_file
    )
    
    # 최종 결과 저장
    if output_file and results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())

