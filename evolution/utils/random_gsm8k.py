from datasets import Dataset, load_dataset


def randomly_sample_gsm8k(dataset_dir, sample_size):
    dataset = load_dataset("gsm8k", "main")

    # Sample
    sampled_dataset = dataset["train"].shuffle().select([i for i in range(sample_size)])

    # Json Lines format
    output_file = dataset_dir + "/random_gsm8k_dataset.jsonl"
    sampled_dataset.to_json(output_file)

    return output_file


if __name__ == '__main__':
    randomly_sample_gsm8k("C:/Files/PycharmProjects/NeuralEvolution/EvoMerge/runs/random_llamas/datasets", 20)
