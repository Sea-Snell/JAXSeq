import json
import tyro
from JaxSeq.utils import convert_path, create_path
import os

def main(
    data_path: str, 
    split: str, 
    output_path: str, 
):
    with open(convert_path(data_path), 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data[split]:
        str_segments = [("BEGINNING OF CONVERSATION:", 0.0)]
        human_keys = sorted([k for k in item.keys() if 'human' in k])
        gpt_keys = sorted([k for k in item.keys() if 'gpt' in k])
        for human_key, gpt_key in zip(human_keys, gpt_keys):
            str_segments.append(("USER: "+item[human_key] + " GPT:", 0.0))
            str_segments.append((item[gpt_key], 1.0))
            str_segments.append(("</s>", 1.0))
        formatted_data.append(str_segments)
    
    output_path = convert_path(output_path)
    create_path(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for example in formatted_data:
            f.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    tyro.cli(main)
