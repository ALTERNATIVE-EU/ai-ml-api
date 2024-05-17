README: INFERENCE MODELS

(required: anaconda)

1. create an environment 1: 
	env for models
	conda create --name alternative python=3.9 
	conda activate alternative
	pip install deepchem rdkit pandas xgboost mordred tensorflow

2. create an environment 2: 
	CDDD env: https://github.com/jrwnter/cddd

3. how to use by terminal:
	* conda activate alternative
	* enter in folder downloaded 
	* enter in ML or AI
	* [command] python inference.py --smiles c1ccccc1O

4. results:
	ML: csv file named results.csv in ML folder
	AI: activity print in terminal




