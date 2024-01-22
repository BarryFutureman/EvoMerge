from setuptools import setup, find_packages

setup(name='evolution', version='0.1', packages=find_packages(),
      install_requires=[
          "trl",
          "peft",
          "gradio",
          "pandas",
          "graphviz",
          "transformers",
          "numpy",
          "evaluate>=0.4.0",
          "jsonlines",
          "numexpr",
          "pybind11>=2.6.2",
          "pytablewriter",
          "rouge-score>=0.0.4",
          "sacrebleu>=1.5.0",
          "scikit-learn>=0.24.1",
          "sqlitedict",
          "tqdm-multiprocess",
          "zstandard",
      ], )
