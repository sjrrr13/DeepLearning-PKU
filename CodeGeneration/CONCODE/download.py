from datasets import load_dataset
import json

data = load_dataset('AhmedSSoliman/CodeXGLUE-CONCODE')
print(data)

for split in data:
    # print(split)
    output_file =  'CodeXGLUE/Text-Code/text-to-code/dataset/concode/{}.jsonl'.format(split)
    with open(output_file, 'w') as f:
        for example in data[split]:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
