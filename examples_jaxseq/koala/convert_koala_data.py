import tyro
import json
from JaxSeq.utils import convert_path, jsonl_load, create_path
from typing import Dict, List, Any, Tuple
import os

def parse_example(
    example: Dict[str, Any], 
    subfield_separator: str=' ',   
) -> List[Tuple[str, float]]:
    str_segments = [("BEGINNING OF CONVERSATION:", 0.0)]
    
    fields = example['fields'].split(',')
    for field in fields:
        if field.startswith('[') and field.endswith(']'):
            # No loss for this field.
            field = field[1:-1]
            mask = 0.0
        else:
            mask = 1.0

        if field == '<|bos|>':
            str_segments.append(('<s>', mask))
        elif field == '<|eos|>':
            str_segments.append(('</s>', mask))
        else:
            subfields = field.split('+')
            text = subfield_separator.join(
                [example[subfield] for subfield in subfields]
            )
            str_segments.append((text, mask))
    
    return str_segments

def main(
    data_path: str, 
    output_path: str, 
):
    with open(convert_path(data_path), 'r') as f:
        data = jsonl_load(f)
    
    parsed_examples = []
    for example in data:
        parsed_examples.append(parse_example(example))
    
    output_path = convert_path(output_path)
    create_path(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for example in parsed_examples:
            f.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    tyro.cli(main)
