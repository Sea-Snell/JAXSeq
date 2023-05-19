import json
import tyro
import random

def main(
    data_path: str, 
    output_path: str, 
):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)

    with open(output_path, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    tyro.cli(main)
