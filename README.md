# IFT6168
IFT6168 Graduate Level class on Causality Final Project

To run this code, ideally it needs to be on a A100 GPU or else it might get stuck or bug. We run it on Compute Canada Narval.

Below are the steps to run the experiments to reproduce results in the Boundless DAS paper.
1. Git clone pyvene repo: https://github.com/stanfordnlp/pyvene/tree/main
2. Install dependencies from this repo and assure that arrow is loaded beforehand with: `module load arrow/14.0.0`: 
    Then you can run : `pip install -r requirements.txt`
3. On Narval, there is only internet connection on its login node, so you need to first download the Alpaca model ("sharpbai/alpaca-7b-merged")
4. In the code at "pyvene/pyvene/models/llama/modelings_intervenable_llama.py" modify the path for the model
5. To allow for gemma model, we also need to add it to the imports of the pyvene repo we cloned at "pyvene/pyvene/__init__.py" and add the following line:
    `from .models.gemma.modelings_intervenable_gemma import create_gemma` 
   For other models, you will need to do the same.
6. Then, to run the code connect to a compute node and activate the virtual environment
7. Then, you can run the default code with: `python code/main.py` or you can use a json config file and use the command `python code/main.py -p config.json` or if you want to change a single hyperparameter without changing the rest with the following `python code/main.py -d "json state string"`
8. To use batch jobs, you can refer to bash script for that and run: `sbatch run_llama.sh`
9. To run sweeps of hyperparameters tuning, you can refer to the following bash scripts: `loop_config.sh`