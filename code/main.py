default_params = {
    "prealign": False,
    "model": "mistral",
    "alignment_sampler": "lower",
    "seed": 42,
    "sample": 10000,
    "batch_size": 64,
    "layer": 5,
    "epochs": 3,
    "gradient_accumulation_steps": 4,
    "temperature_start": 50.0,
    "temperature_end": 0.1,

}

def defaults(dictionary, dictionary_defaults):
    for key, value in dictionary_defaults.items():
        if key not in dictionary:
            dictionary[key] = value
        else:
            if isinstance(value, dict) and isinstance(dictionary[key], dict):
                dictionary[key] = defaults(dictionary[key], value)
            elif isinstance(value, dict) or isinstance(dictionary[key], dict):
                raise ValueError("Given dictionaries have incompatible structure")
    return dictionary

import sys
sys.path.append("/home/qinjerem/scratch/IFT6168/pyvene")

import time
import pyvene
import torch
import os
import wandb
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tutorial_price_tagging_utils import (
    factual_sampler,
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
)

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import create_llama, create_gemma, create_gpt_neo, create_gpt2, create_gpt2_classifier, create_mistral
from pyvene import set_seed, count_parameters


def experiment(params):
    from prealign_task import asses_prealign_task
    from create_data import create_data
    from boundless_das_position_config import simple_boundless_das_position_config
    from metrics import compute_metrics
    from loss import calculate_loss

    torch.cuda.empty_cache()
    os.environ['WANDB_MODE'] = 'offline'

    prealign = params["prealign"]
    model = params["model"]
    alignment_sampler = params["alignment_sampler"]
    seed = params["seed"]
    sample = params["sample"]
    batch_size = params["batch_size"]
    layer = params["layer"]
    epochs = params["epochs"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]
    temperature_start = params["temperature_start"]
    temperature_end = params["temperature_end"]

    set_seed(seed)
    print("Initializing Experiment")
    wandb.init(
        project="causality",
        config=params,
        name=f"{model}_seed{seed}_layer{layer}_bs{batch_size}_{alignment_sampler}"
    )

    if model == "llama":
        config, tokenizer, model = create_llama()
    elif model == "gemma":
        config, tokenizer, model = create_gemma()
    elif model == "gpt2":
        config, tokenizer, model = create_gpt2_classifier()
    else:
        config, tokenizer, model = create_mistral()
    _ = model.to("cuda")  
    # _ = model.eval() 
    print("config")
    print(config)

    print("Task alignment check")
    if prealign == True:
        asses_prealign_task(model, tokenizer)
    else:
        print("Skip prealigning")

    ###################
    # data loaders
    ###################
    print("Create Dataloaders")
    train_dataloader, eval_dataloader, test_dataloader = create_data(sample, tokenizer, batch_size, alignment_sampler)

    print("Intervention setup")
    config = simple_boundless_das_position_config(type(model), "block_output", layer)
    print("intervention model setup")
    intervenable = IntervenableModel(config, model, use_fast=True)
    print("intervetion model disable gradients")
    intervenable.set_device("cuda")
    intervenable.disable_model_gradients()

    print("Hyperparams init")
    t_total = int(len(train_dataloader) * 3)
    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        optimizer_params += [{"params": v[0].intervention_boundaries, "lr": 1e-2}]
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    total_step = 0
    target_total_step = len(train_dataloader) * epochs
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, target_total_step)
        .to(torch.bfloat16)
        .to("cuda")
    )
    intervenable.set_temperature(temperature_schedule[total_step])
    print("Training")
    intervenable.model.train()  
    print("llama trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())
    train_iterator = trange(0, int(epochs), desc="Epoch")
    
    #Assign which token to swar
    if model == "llama":
        token_swap = 80
    elif model == "mistral":
        token_swap = 77

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": token_swap},
            )
            eval_metrics = compute_metrics(
                [counterfactual_outputs.logits], [inputs["labels"]]
            )

            # loss and backprop
            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], intervenable)
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})
            wandb.log({"loss": loss, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    intervenable.set_temperature(temperature_schedule[total_step])
            total_step += 1
    # evaluation on the test set
    print("Evaluation")
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": token_swap},
            )
            eval_labels += [inputs["labels"]]
            eval_preds += [counterfactual_outputs.logits]
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    print(eval_metrics)
    wandb.log({"eval acc": eval_metrics["accuracy"]})
    wandb.finish()
    print("Done")

if __name__ == "__main__":
    import json
    import argparse
    import time
    from datetime import datetime

    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", type=str, help="JSON params file")
    parser.add_argument("--direct", "-d", type=str, help="JSON state string")

    arguments = parser.parse_args()

    if arguments.direct is not None:
        params = json.loads(arguments.direct)
    elif arguments.params is not None:
        with open(arguments.params) as file:
            params = json.load(file)
    else:
        params = {}

    params = defaults(params, default_params)
    experiment(params)

    print("Done")
    end = time.time()
    total_time = (end - start) / 60

    print(f'Total time: {total_time} min')