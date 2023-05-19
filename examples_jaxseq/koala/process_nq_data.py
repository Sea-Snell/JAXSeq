import json
from JaxSeq.utils import convert_path
import tyro
from typing import List
from tqdm.auto import tqdm

def parse_example(example):
    answers = []
    for annotation in example['annotations']:
        if annotation['yes_no_answer'] != 'NONE':
            answers.append(annotation['yes_no_answer'].lower())
        for answer in annotation['short_answers']:
            answers.append(' '.join(map(lambda x: x['token'], example['document_tokens'][answer['start_token']:answer['end_token']])))
    question = example['question_text']
    if len(answers) == 0:
        return None
    return {'in_text': question, 'out_text': answers[0], 'answers': answers, 'example': example}

def main(nq_paths: List[str]):

    all_data = []
    for nq_path in nq_paths:
        print(nq_path)
        with open(nq_path, 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                example = parse_example(data)
                if example is None:
                    continue
                all_data.append(example)

# need to add prompt, save data

if __name__ == "__main__":
    tyro.cli(main)