# EvoMerge - Neuroevolution for Large Language Models
### It would be fantastic if someone could endorse me for arXiv so I can publish a paper. The endorsement code is: TJWH9Z

## Quick Start:
#### 1. Installation
```
git clone https://github.com/BarryFutureman/EvoMerge
cd EvoMerge
pip install .
# Or pip install -e .
```
Note that I'm using a docker for text generation webui, the setup.py file may not contain all necessary packages (Need to verify this).
#### 2. Simulation Folder
MistralEvolve is an example of a simulation folder. The population folder contains all the models, you need to download the initial population there (see download_model.py for an example to download models from huggingface).
The config folder has the yaml config file, and the DNAs folder contains information about each generation.

#### 3. Run Simulation
Once the folder is set up, go into run_simualtion.py, scroll all the way down to configure the path. I should work on a less ridiculous way to do this...
```
cd evolution
python run_simualtion.py
```
You can then visualize the simulation using
```
sudo apt-get install graphviz  # or some other way to install graphviz
python run_control_panel.py
```

## Updates:
- 2024/01/29 Making the repo public. Will work on it more during reading work (Feb 19th).
