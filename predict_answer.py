import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
# from datasets import load_dataset
import json
from multiprocessing import Pool
from functools import partial
import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score
import ast

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def normalize_prediction(prediction):
    # 리스트인 경우
    if isinstance(prediction, list):
        # 리스트의 첫 번째 요소가 문자열로 된 리스트인 경우 (예: ["['Japan', 'Ireland']"])
        if len(prediction) > 0 and isinstance(prediction[0], str) and prediction[0].startswith('['):
            try:
                return ast.literal_eval(prediction[0])
            except:
                return prediction
        return prediction
    # 문자열인 경우
    elif isinstance(prediction, str):
        # 문자열이 리스트 형태인 경우 (예: "['Japan', 'Ireland']")
        if prediction.startswith('['):
            try:
                return ast.literal_eval(prediction)
            except:
                return prediction.split('\n')
        return prediction.split('\n')
    return prediction

def eval_hit(prediction, answer):
    if isinstance(answer, str):
        answer = ast.literal_eval(answer)

    prediction = normalize_prediction(prediction)
    
    # prediction이 리스트인 경우 각 요소에 대해 확인
    if isinstance(prediction, list):
        for pred in prediction:
            for a in answer:
                if match(str(pred), str(a)):
                    return 1
        return 0
    
    # prediction이 문자열인 경우
    for a in answer:
        if match(prediction, str(a)):
            return 1
    return 0

def eval_hit_at_1(prediction, answer):
    if isinstance(answer, str):
        answer = ast.literal_eval(answer)
    
    prediction = normalize_prediction(prediction)
    
    # prediction이 리스트인 경우 첫 번째 값만 사용
    if isinstance(prediction, list):
        if len(prediction) > 0:
            prediction = str(prediction[0])
        else:
            return 0
    # prediction이 문자열인 경우 첫 번째 줄만 사용
    elif isinstance(prediction, str):
        prediction = prediction.split('\n')[0]
    
    for a in answer:
        if match(prediction, str(a)):
            return 1
    return 0

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            # id = data['id']
            prediction = data['prediction']
            answer = data['ground_truth']
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
  
                prediction_str = ' '.join(prediction)
                hit = eval_hit(prediction_str, answer)
                hit_list.append(hit)
                f2.write(json.dumps({'prediction': prediction, 'ground_truth': answer, 'hit': hit}) + '\n')
            else:
                hit = eval_hit(prediction, answer)
                hit_list.append(hit)
                f2.write(json.dumps({'prediction': prediction, 'ground_truth': answer,'hit': hit}) + '\n')
    

    result_str = " Hit: " + str(sum(hit_list) * 100 / len(hit_list))
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        return open(path, "w")
    else:
        return open(path, "a")



def prediction(data, model):
    question = data["instruction"]
    answer = data["output"]
    prediction = model.generate_sentence(question)
    if prediction is None:
        return None
    result = {
        "question": question,
        "prediction": prediction,
        "ground_truth": answer
    }
    return result


def main(args,LLM):
    rule_postfix = "no_rule"
    
    # 입력 파일 경로를 파라미터로 받음
    input_file = args.input_file
    with open(input_file, "r") as json_file:
        dataset = json.load(json_file)
    print("Load dataset from finished")

    output_dir = args.predict_dir
    print("Save results to: ", output_dir) # datasets/TimeQuestions/result
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        print("Prepare pipline for inference...")
        model.prepare_for_inference()

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout = get_output_file(output_file, force=args.force)

    # 중간 결과를 저장하기 위한 리스트
    temp_results = []
    total_samples = len(dataset)
    checkpoint_interval = total_samples // 20  # 5% 간격으로 변경

    if args.n > 1:
        with Pool(args.n) as p:
            for i, res in enumerate(tqdm(
                p.imap(
                    partial(
                        prediction,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            )):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                    temp_results.append(res)
                    
                    # 5%마다 중간 결과 출력
                    if (i + 1) % checkpoint_interval == 0:
                        progress = ((i + 1) / total_samples) * 100
                        # 현재까지의 결과로 Hit rate 계산
                        hit_count = sum(1 for r in temp_results if eval_hit(r['prediction'], r['ground_truth']))
                        current_hit_rate = (hit_count / len(temp_results)) * 100
                        # Hit@1 계산
                        hit_at_1_count = sum(1 for r in temp_results if eval_hit_at_1(r['prediction'], r['ground_truth']))
                        current_hit_at_1_rate = (hit_at_1_count / len(temp_results)) * 100
                        print(f"\nProgress: {progress:.1f}%", flush=True)
                        print(f"Current Hit: {current_hit_rate:.2f}%", flush=True)
                        print(f"Current Hit@1: {current_hit_at_1_rate:.2f}%", flush=True)
                        sys.stdout.flush()
    else:
        for i, data in enumerate(tqdm(dataset)):
            res = prediction(data, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                temp_results.append(res)
                
                # 5%마다 중간 결과 출력
                if (i + 1) % checkpoint_interval == 0:
                    progress = ((i + 1) / total_samples) * 100
                    # 현재까지의 결과로 Hit rate 계산
                    hit_count = sum(1 for r in temp_results if eval_hit(r['prediction'], r['ground_truth']))
                    current_hit_rate = (hit_count / len(temp_results)) * 100
                    # Hit@1 계산
                    hit_at_1_count = sum(1 for r in temp_results if eval_hit_at_1(r['prediction'], r['ground_truth']))
                    current_hit_at_1_rate = (hit_at_1_count / len(temp_results)) * 100
                    print(f"\nProgress: {progress:.1f}%", flush=True)
                    print(f"Current Hit: {current_hit_rate:.2f}%, count : {hit_count}", flush=True)
                    print(f"Current Hit@1: {current_hit_at_1_rate:.2f}%, count : {hit_at_1_count}", flush=True)
                    sys.stdout.flush()
                    if total_samples > 3000 and int(progress) == 5:
                        break
        
    fout.close()
    
    # 최종 결과 계산 및 출력
    hit_count = sum(1 for r in temp_results if eval_hit(r['prediction'], r['ground_truth']))
    hit_rate = (hit_count / len(temp_results)) * 100
    hit_at_1_count = sum(1 for r in temp_results if eval_hit_at_1(r['prediction'], r['ground_truth']))
    hit_at_1_rate = (hit_at_1_count / len(temp_results)) * 100
    result_str = f" Hit: {hit_rate:.2f}% (count: {hit_count}/{len(temp_results)})\n Hit@1: {hit_at_1_rate:.2f}% (count: {hit_at_1_count}/{len(temp_results)})"
    print(result_str)
    
    # 평가 결과 저장
    eval_result_path = output_file.replace('predictions.jsonl', 'eval_result.txt')
    with open(eval_result_path, 'w') as f:
        f.write(result_str)
        
    # detailed_eval_result.jsonl 생성
    print("\n상세 평가 결과 생성 중...")
    eval_result(output_file, cal_f1=True)
    print("상세 평가 결과 생성 완료")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--d", "-d", type=str, default="TimeQuestions")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_dir", type=str, default="results")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="llama",
    )
    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()
    main(args, LLM)
