from datasets import load_dataset
from transformers import AutoTokenizer


def belle_open_source_500k(data_file, tokenizer, max_len):
    def tokenize(input_text, target_text):
        input_text_result = tokenizer(
            input_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        target_text_result = tokenizer(
            target_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        if target_text_result["input_ids"][0] == tokenizer.bos_token_id:
            target_text_result["input_ids"] = target_text_result["input_ids"][1:]
            target_text_result["attention_mask"] = target_text_result["attention_mask"][1:]

        if target_text_result['input_ids'][-1] != tokenizer.eos_token_id:
            target_text_result["input_ids"].append(tokenizer.eos_token_id)
            target_text_result["attention_mask"].append(1)

        if input_text_result['input_ids'][0] != tokenizer.bos_token_id:
            input_text_result["input_ids"] = [tokenizer.bos_token_id] + input_text_result["input_ids"]
            input_text_result["attention_mask"].append(1)

        if input_text_result['input_ids'][-1] == tokenizer.eos_token_id:
            input_text_result["input_ids"] = input_text_result["input_ids"][:-1]
            input_text_result["attention_mask"] = input_text_result["attention_mask"][:-1]

        result = {'input_ids': input_text_result["input_ids"] + target_text_result["input_ids"],
                  'attention_mask': input_text_result["attention_mask"] + target_text_result["attention_mask"],
                  'loss_mask': [0] * len(input_text_result['input_ids']) + [1] * len(target_text_result["input_ids"])}

        if len(result["input_ids"]) > max_len:
            result["input_ids"] = result["input_ids"][:max_len - 1] + [tokenizer.eos_token_id]
            result["attention_mask"] = result["attention_mask"][:max_len]
            result['loss_mask'] = result['loss_mask'][:max_len]

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        instruction = data_point['instruction']
        input_text = data_point["input"]
        input_text = "Human: " + instruction + input_text + "\n\nAssistant: "
        target_text = data_point["output"]
        tokenized_full_prompt = tokenize(input_text, target_text)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_file)["train"]
    data = data.map(generate_and_tokenize_prompt, num_proc=8)
    data = data.remove_columns(['instruction', 'output', 'input'])
    return data


if __name__ == "__main__":
    tk = AutoTokenizer.from_pretrained("./model_config",
                                       trust_remote_code=True)
    ds = belle_open_source_500k("./data/Belle_open_source_200.json", tk, 512)
    print(ds[1])
    print(tk.decode(ds[1]['input_ids']))
