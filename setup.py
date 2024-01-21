from setuptools import setup, find_packages

setup(name='evolution', version='0.1', packages=find_packages(),
      install_requires=[
          "trl",
          "peft",
          "gradio",
          "pandas",
          "graphviz",
          "transformers",
          "numpy"
    ],)
