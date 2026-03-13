import torch
import torch.utils
from argparse import ArgumentParser
import pickle as pkl
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path

def filter_nounphrases(nps):
    if len(nps) == 0:
        return []
    phrases = {}
    for np in nps:
        if np not in phrases and len(np) > 1:
            phrases[np] = 1
    return [key for key in phrases]


def post_process_nounphrases(raw_outputs: str, org_sentence: str):
    lines = raw_outputs.splitlines()
    outputs = []
    for item in lines:
        line = item.strip()
        spans = line.split(". ")
        if len(spans) != 2:
            continue
        
        try:
            # check if this is an indexed item in the list
            t = int(spans[0]) 
        except:
            continue
        
        if len(spans[1]) > len(org_sentence):
            continue

        if "(" in spans[1] or ")" in spans[1] or "excluded" in spans[1]:
            continue
        
        outputs.append(spans[1].strip())
    outputs = filter_nounphrases(outputs)
    return outputs


def get_model(device=None):
    # create model
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", 
        device_map=device if device is not None else "cpu", 
        torch_dtype="auto", 
        # trust_remote_code=True,
        _attn_implementation='flash_attention_2' if device is not None else "eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    return model, tokenizer


def get_chat_message(sentence):
    query_template = "Can you extract the unique noun phrases of the main subjects, except generic locations such as home or street or school, in the following sentence: {SENTENCE}? Return only the noun phrases in an indexed list. Do not add explanation, justitication, note or remark. Ignore phrases such as [stock photo] or [news photo]."
    query = query_template.replace("{SENTENCE}", sentence)
    message = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you extract the unique noun phrases of the main subjects, except generic locations such as home or street or school, in the following sentence: A woman wearing red dress stands next to a fireplace likely at a university? Return only the noun phrases in an indexed list. Do not add explanation, justitication, note or remark. Ignore phrases such as [stock photo] or [news photo]."},
        {"role": "assistant", "content": "1. A woman\n2. red dress\n3. fireplace"},
        {"role": "user", "content": query},
    ]
    return message


def get_chat_messages(sentences):
    messages = [get_chat_message(sentence) for sentence in sentences]
    return messages


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, metadata: str|dict = "./data_dir/images/cc3m_images.pkl"):
        # load dataset
        if isinstance(metadata, str):
            with open(metadata, "rb") as fp:
                data = pkl.load(fp)
        elif isinstance(metadata, dict):
            data = metadata
        else:
            raise ValueError
        
        self.captions = {}
        self.keys = []
        for k, v in data.items():
            self.captions[int(k)] = v["captions"]["shortLLA_captions"]
            self.keys.append(int(k))
        
        self.keys.sort()

    
    def __len__(self):
        return len(self.keys)


    def __getitem__(self, index):
        index = self.keys[index]
        caption = self.captions[index]
        return index, caption


class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """ 
        batch is a list of (index-caption) pairs
        """
        indices = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        # format captions -> chat messages
        messages = get_chat_messages(captions)
        # generate text prompts
        prompts = [self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        # tokenize text prompts
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", max_length=1024, truncation=True)

        return inputs, indices, captions


def run_batch(model, tokenizer, device, batch: list[str]):
    generation_args = {
        "max_new_tokens": 100,
        #"temperature": 0.0,
        "do_sample": False,
    }

    model_inputs, indices, captions = batch
    model_inputs = model_inputs.to(device)
    
    generate_ids = model.generate(**model_inputs, 
        eos_token_id=tokenizer.eos_token_id, 
        **generation_args
    )

    # remove input tokens & decode
    generate_ids = generate_ids[:, model_inputs['input_ids'].shape[1]:]
    model_outputs = tokenizer.batch_decode(generate_ids,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)
    
    outputs = {}
    # post-process nounphrases
    for index, model_output, cap in zip(indices, model_outputs, captions):
        nps = post_process_nounphrases(model_output, cap)
        outputs[index] = nps

    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("--metadata-pkl-file", type=str, default="./data_dir/images/cc3m_images.pkl", help="default path for processed metadata")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite-metadata-file", default=False, action="store_true")

    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if args.gpu is not None and args.gpu >= 0 else "cpu"
    metadata_file = Path(args.metadata_pkl_file)
    if args.overwrite_metadata_file:
        output_file = metadata_file
    else:
        output_file = metadata_file.parent / f"{metadata_file.stem}_nounphrases{metadata_file.suffix}"

    # create model
    model, tokenizer = get_model(device=device)

    # load metadata
    with open(metadata_file, "rb") as fp:
        metadata = pkl.load(fp)

    # create dataset
    dataset = CaptionDataset(metadata=metadata)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=DataCollator(tokenizer=tokenizer),
                                              shuffle=False,
                                              num_workers=args.workers,
                                              drop_last=False)
    
    for i, batch in enumerate(tqdm(data_loader)):
        outputs = run_batch(model, tokenizer, device, batch)

        for k, v in outputs.items():
            metadata[str(k)]["nounphrases"] = v

        if i % 100 == 99:
            # save every 100 batches
            with open(output_file, "wb") as fp:
                pkl.dump(metadata, fp)
    
    # last save to output_pkl
    with open(output_file, "wb") as fp:
        pkl.dump(metadata, fp)
    
if __name__ == "__main__":
    main()
