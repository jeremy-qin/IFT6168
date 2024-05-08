from datasets import Dataset
from torch.utils.data import DataLoader
from tutorial_price_tagging_utils import (
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
    midpoint_alignment_sampler,
    bracket_alignment_sampler
)

def create_data(sample, tokenizer, batch_size, alignment_sampler):
    print(alignment_sampler)
    if alignment_sampler == "lower":
        raw_data = bound_alignment_sampler(
            tokenizer, sample, [lower_bound_alignment_example_sampler]
        )
    elif alignment_sampler == "midpoint":
        raw_data = midpoint_alignment_sampler(tokenizer, sample)
    elif alignment_sampler == "bracket":
        raw_data = bracket_alignment_sampler(tokenizer, sample)
    else:
        raise NotImplementedError("This alignment sampler is not implemented.")

    raw_train = (
        raw_data[0][:8000],
        raw_data[1][:8000],
        raw_data[2][:8000],
        raw_data[3][:8000],
    )
    raw_eval = (
        raw_data[0][8000:9000],
        raw_data[1][8000:9000],
        raw_data[2][8000:9000],
        raw_data[3][8000:9000],
    )
    raw_test = (
        raw_data[0][9000:],
        raw_data[1][9000:],
        raw_data[2][9000:],
        raw_data[3][9000:],
    )
    train_dataset = Dataset.from_dict(
        {
            "input_ids": raw_train[0],
            "source_input_ids": raw_train[1],
            "labels": raw_train[2],
            "intervention_ids": raw_train[3],  # we will not use this field
        }
    ).with_format("torch")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    eval_dataset = Dataset.from_dict(
        {
            "input_ids": raw_eval[0],
            "source_input_ids": raw_eval[1],
            "labels": raw_eval[2],
            "intervention_ids": raw_eval[3],  # we will not use this field
        }
    ).with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
    )
    test_dataset = Dataset.from_dict(
        {
            "input_ids": raw_test[0],
            "source_input_ids": raw_test[1],
            "labels": raw_test[2],
            "intervention_ids": raw_test[3],  # we will not use this field
        }
    ).with_format("torch")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_dataloader, eval_dataloader, test_dataloader