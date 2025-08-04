import os
import csv
import pickle
import time
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
from retrieval_temporal import Retrieval
import torch
from process_test_data import batch_extract_entities, get_device
from transformers import AutoTokenizer, AutoModelForTokenClassification

os.environ['OPENAI_BASE_URL'] = 'YOUR_OPENAI_URL'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'

#retrieval
# def retrieve(d,model_name, question_list, triple_list):
#     retriever = Retrieval(d,model_name, question_list, triple_list)
#     distances, corpus_ids = retriever.compute_similarity(n=15)
#     fact_list = retriever.get_result(distances, corpus_ids, question_list,re_rank=False)
def retrieve(d, model_name, question_list, triple_list, question_to_slot, num_subgraphs=1000, answer_slot_method="masking", topic_entity_list=None, answer_types=None):
    retriever = Retrieval(d, model_name, question_list, triple_list, question_to_slot, answer_slot_method=answer_slot_method, topic_entity_list=topic_entity_list)
    # num_subgraphs보다 더 많은 후보를 검색하여 재순위화에 사용
    # search_size = min(2000, num_subgraphs * 2)  # 최소한 num_subgraphs의 2배는 검색
    search_size = num_subgraphs
    # search_size = 15
    distances, corpus_ids = retriever.compute_similarity(n=search_size)
    fact_list = retriever.get_result(distances, corpus_ids, question_list, re_rank=False, answer_types=answer_types)
    return fact_list

#rewrite
def gpt_chat_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            client = OpenAI()
            # client = Client()
            response = client.chat.completions.create(**kwargs)
            if type(response) == str:
                return response_parser(response)
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(e)
            # print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def response_parser(response):
    response = response.strip().split("data: ")[1:]
    result = ''
    for r in response:
        if r == '[DONE]':
            break
        delta = json.loads(r)['choices'][0]['delta']
        if 'content' in delta:
            result += delta['content']
    return result

# def rewrite(fact_list,question_list):
#     # 이미 저장된 rewrite.json 파일에서 rewrite_question만 추출
#     with open('/home/bys3158/TimeR4/datasets/MultiTQ/rewite.json', 'r', encoding='utf-8') as file:
#         rewrite_data = json.load(file)
#         # rewrite_question만 추출하여 리스트로 반환
#         return [item['rewrite_question'] for item in rewrite_data]

def read_entity2id(file_path):
    entity_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  
                entity, entity_id = line.split('\t')
                entity_dict[entity_id] = entity
    return entity_dict

def main():
    triple_list = []
    with open(args.triplet_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplets = line.strip().replace("_", " ").split('\t')
            triple_list.append(triplets)
    with open(args.question_path, 'r') as file:
        question_json = json.load(file)
    question_list = [q['question'] for q in question_json]
    question_to_slot = None
    # # answer_slot 정보 로드 (test_tear_path에서 가져옴)
    # with open(args.test_tear_path, 'r', encoding='utf-8') as file:
    #     vote_data = json.load(file)
    # question_to_slot = {item['question']: item['answer_slot'] for item in vote_data}

    # # answer_type 리스트 생성
    # answer_types = [item['answer_type'].capitalize() for item in vote_data]

    # # topic_entity 정보 로드 (test_tear_vote+entity.json에서 가져옴)
    # with open('datasets/MultiTQ/questions/test_tear_vote+entity.json', 'r', encoding='utf-8') as file:
    #     entity_data = json.load(file)
    # question_to_topic_entity = {item['question']: item['topic_entity'] for item in entity_data}

    # # topic_entity_list 생성 (질문 순서에 맞게 answer_slot과 topic_entity를 모두 포함)
    # topic_entity_list = [
    #     {"answer_slot": item["answer_slot"], "topic_entity": question_to_topic_entity.get(item["question"], None)}
    #     for item in vote_data
    # ]

    with open('datasets/MultiTQ/rewrite.json', 'r', encoding='utf-8') as file:
        rewrite_data = json.load(file)
        rewrite_questions = [item['rewrite_question'] for item in rewrite_data]
    
    # question_json에 rewrite_question 추가
    for i, q in enumerate(question_json):
        q['rewrite_question'] = rewrite_questions[i]

    # topic_entity 관련 NER/추출 부분 완전 제거
    # vote_data, topic_entity_list 등도 제거

    #time retrival
    # fact_list = retrieve(args.d, args.retrieve_name, question_list, triple_list, question_to_slot, args.num_subgraphs) 
    # fact_list = retrieve(
    #     args.d, args.retrieve_name, rewrite_questions, triple_list, question_to_slot, args.num_subgraphs,
    #     answer_slot_method=args.answer_slot_method, topic_entity_list=topic_entity_list, answer_types=answer_types
    # )
    fact_list = retrieve(
        args.d, args.retrieve_name, rewrite_questions, triple_list, question_to_slot, args.num_subgraphs,
        answer_slot_method=args.answer_slot_method, topic_entity_list=None, answer_types=None
    )
    print("retrieval done, start saving...")
    print(f"fact_list length: {len(fact_list)}")
    print(f"fact_list sample: {fact_list[:2]}")
    # get_result()에서 top_k=15로 fact/triple을 자르기
    top_k = 15
    for result in fact_list:
        if 'fact' in result:
            result['fact'] = result['fact'][:top_k]
        if 'triple' in result:
            result['triple'] = result['triple'][:top_k]
    # generate prompt
    assert len(question_list)==len(fact_list)
    
    # .pt 파일로 저장
    output_file = f'datasets/MultiTQ/results/{args.output_filename}.pt'
    
    # 기존 파일이 있는지 확인
    if os.path.exists(output_file):
        print(f"\n기존 파일 {output_file}을 사용합니다.")
        results = torch.load(output_file)
    else:
        print(f"\n새로운 파일 {output_file}을 생성합니다.")
        results = []
        for i in range(len(fact_list)):
            question = question_json[i]['rewrite_question']  # 재작성된 질문 사용
            result = {
                'question': question,
                'quadruples': fact_list[i]['triple'][:args.num_subgraphs],  # 지정된 개수의 쿼드러플
            }
            results.append(result)
        print(f"result_list length: {len(results)}")
        print(f"result_list sample: {results[:2]}")
        # 전체 저장
        print(f"saving all {len(results)} results to {output_file}")
        torch.save(results, output_file)
        print(f"서브그래프가 {output_file}에 저장되었습니다.")
        print(f"각 질문당 {args.num_subgraphs}개의 쿼드러플이 저장되었습니다.")

    result_list = []
    print("\n결과 포맷팅 중...")
    try:
        for i in tqdm(range(len(fact_list)), desc="질문 처리"):
            question = question_json[i]['rewrite_question']  # 재작성된 질문 사용
            fact = fact_list[i]['fact']
            triple_id = fact_list[i]['triple']
            sentences = []
            answers = question_json[i]['answers']
            text = f"Based on the facts, please answer the given question. Keep the answer as simple as possible and return all the possible answers as a list.\n Facts:{fact}\nQuestion:\n {question}?" 
            while len(text) > 1024:
                fact.pop()   
                triple_id.pop()
                text = f"Based on the facts, please answer the given question. Keep the answer as simple as possible and return all the possible answers as a list.\n Facts:{fact}\nQuestion:\n {question}?"   

            if args.type == "train":
                formatted_data = {
                    "instruction":text,
                    "output":str(answers),
                    "input":"",
                    "embedding_ids":triple_id
                }
            else:
                formatted_data = {
                    "text": text,
                    "answers": str(answers),
                    "question":question,
                    "embedding_ids":triple_id
                }
            result_list.append(formatted_data)
        print(f"result_list length: {len(result_list)}")
        print(f"result_list sample: {result_list[:2]}")
        print(f"saving all {len(result_list)} result_list to {args.output_path}")
        # 전체 저장
        with open(args.output_path, "w", encoding='utf-8') as json_file:
            json_str = json.dumps(result_list, ensure_ascii=False, indent=4)
            json_file.write(json_str)
        print(f"저장 완료! 총 {len(result_list)}개의 결과가 저장되었습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print(f"현재까지 처리된 결과 수: {len(result_list)}")
        print("오류 발생 지점의 상태:")
        print(f"- 현재 처리 중이던 질문 인덱스: {i}")
        print(f"- fact_list 길이: {len(fact_list)}")
        print(f"- question_json 길이: {len(question_json)}")
        raise  # 오류를 다시 발생시켜 작업 중단

    # 필요에 따라 vote_data를 활용

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--triplet_path", type=str)
    argparser.add_argument("--question_path", type=str)
    argparser.add_argument("--rewrite_output_path", type=str)
    argparser.add_argument("--retrieve_name", type=str, default='time_aware_model_tp')
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument("--d", default='time')
    argparser.add_argument("--type", default='train')
    argparser.add_argument("--num_subgraphs", type=int, default=1000, help='각 질문당 저장할 쿼드러플 개수')
    argparser.add_argument("--output_filename", type=str, default='question_quadruples_answer_slot_mask', help='저장할 파일명')
    argparser.add_argument("--answer_slot_method", type=str, default='masking', choices=['masking','penalty','original'], help='answer slot 처리 방식')
    argparser.add_argument("--test_tear_path", type=str, default='datasets/MultiTQ/questions/test_tear_strict_positive_output.json', help='test_tear.json 경로')
    args = argparser.parse_args()
    main()