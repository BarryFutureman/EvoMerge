# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv

import datasets


_CITATION = """\
@article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
"""

_DESCRIPTION = """\
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
"""

_HOMEPAGE = "https://github.com/hendrycks/test"

_URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/data.tar"

_SUBJECTS = [
    "all",
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class Mmlu(datasets.GeneratorBasedBuilder):
    """Measuring Massive Multitask Language Understanding, consisting of 57 tasks"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=sub, version=datasets.Version("1.0.0"), description=f"MMLU Subject {sub}"
        )
        for sub in _SUBJECTS
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "subject": datasets.Value("string"),
                "choices": datasets.features.Sequence(datasets.Value("string")),
                "answer": datasets.features.ClassLabel(num_classes=4, names=["A", "B", "C", "D"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archive = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split("auxiliary_train"),
                gen_kwargs={
                    "iter_archive": dl_manager.iter_archive(archive),
                    "split": "auxiliary_train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"iter_archive": dl_manager.iter_archive(archive), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "iter_archive": dl_manager.iter_archive(archive),
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("dev"),
                gen_kwargs={
                    "iter_archive": dl_manager.iter_archive(archive),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, iter_archive, split):
        """Yields examples as (key, example) tuples."""
        n_yielded_files = 0
        for id_file, (path, file) in enumerate(iter_archive):
            if f"data/{split}/" in path:
                if split == "auxiliary_train" or f"{self.config.name}_{split}.csv" in path or self.config.name == "all":
                    subset = path.split("/")[-1].rsplit("_",1)[0] if split != "auxiliary_train" else ""
                    n_yielded_files += 1
                    lines = (line.decode("utf-8") for line in file)
                    reader = csv.reader(lines)
                    for id_line, data in enumerate(reader):
                        yield f"{id_file}_{id_line}", {"question": data[0], "choices": data[1:5], "answer": data[5], "subject": subset}
                    if (n_yielded_files == 8 or split != "auxiliary_train") and self.config.name != "all":
                        break


def randomly_sample_hellaswag(dataset_dir, sample_size):
    builder = Mmlu(config_name="all")

    # Download the dataset
    builder.download_and_prepare()
    dataset = builder.as_dataset()

    # Sample
    sampled_dataset = dataset["validation"].shuffle().select([i for i in range(sample_size)])

    # Json Lines format
    output_file = dataset_dir + "/random_mmlu_dataset.jsonl"
    sampled_dataset.to_json(output_file)

    return output_file


if __name__ == '__main__':
    randomly_sample_hellaswag("C:/Files/PycharmProjects/NeuralEvolution/EvoMerge/runs/random_llamas/datasets", 20)