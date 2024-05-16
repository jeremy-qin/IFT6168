import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from tutorial_price_tagging_utils import (
    factual_sampler
)

def asses_prealign_task(model, tokenizer):

    raw_prealign = factual_sampler(tokenizer, 5000, game="pricing_tag")
    prealign_dataset = Dataset.from_dict(
        {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
    )
    prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
    prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)

    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id = tokenizer.convert_tokens_to_ids("No")

    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(prealign_dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
            )
            
            logits = outputs.logits[:, -1]

            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]

            pred_test_labels = torch.where(yes_logits > no_logits, yes_id, no_id)

            actual_test_labels = inputs["labels"][:, -1]

            correct_labels = actual_test_labels == pred_test_labels

            total_count += len(correct_labels)
            correct_count += correct_labels.sum().tolist()

            if step == 0: 
                decoded_inputs = [tokenizer.decode(ids) for ids in inputs['input_ids']]
                print(f"Sample decoded inputs: {decoded_inputs}")

                print(f"Sample Yes logits: {yes_logits}")
                print(f"Sample No logits: {no_logits}")
                print(f"Sample Predicted Labels: {pred_test_labels}")
                print(f"Sample Actual Labels: {actual_test_labels}")

    current_acc = round(correct_count / total_count, 2)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")
