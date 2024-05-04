import torch
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from tutorial_price_tagging_utils import (
    factual_sampler
)

def asses_prealign_task(model, tokenizer):
    # if model == "llama":
    #     config, tokenizer, model = create_llama()
    # elif model == "gemma":
    #     config, tokenizer, model = create_gemma()
    # _ = model.to("cuda")  
    # _ = model.eval()  

    raw_prealign = factual_sampler(tokenizer, 5000, game="pricing_tag")
    prealign_dataset = Dataset.from_dict(
        {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
    )
    prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
    prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)

    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(prealign_dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)

            # aligning forward!
            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
            )

            actual_test_labels = inputs["labels"][:, -1]
            pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

            correct_labels = actual_test_labels == pred_test_labels

            total_count += len(correct_labels)
            correct_count += correct_labels.sum().tolist()
    current_acc = round(correct_count / total_count, 2)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")
