import pandas as pd

import pickle
import os
import argparse
import subprocess
import sys

# rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def check_smiles(smiles):
    try: Chem.MolFromSmiles(smiles)
    except: 
        print("invalid smiles.")
        sys.exit(1)

def ad_evaluation(smiles:str, radius=3, num_bits=1024):
    """
    check similarity between compounds in dataset and target using morgan fingerprint as descriptors
    """
    fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits=num_bits)
    ad_results = {}
    path_folder = os.path.join(os.getcwd(), "PipelineAlternative_clinicaldata", "ML_apical", "data")
    path_folder_endpoint = os.path.join(path_folder, f'train_apical.csv')
    # name = folder.split("_")[0]
    data = pd.read_csv(path_folder_endpoint)
    smiles_list = list(data['smiles'])
    counter = 0
    for smi in smiles_list:
        print(f"[DEBUGGING]: {smi}")
        try: 
            fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=num_bits)
            if counter < 3: 
                if DataStructs.DiceSimilarity(fp1, fp2) > 0.3: 
                    counter += 1
        except:
            print(f'error molecule {smi}')
            continue
        
        if counter < 3: ad_results["apical_data"] = 'out AD'
        else: ad_results["apical_data"] = 'in AD'

    return pd.DataFrame(ad_results, index=[smiles])

def import_models():
    "import models find in models_apical folder"

    path = os.path.join(os.getcwd(), "PipelineAlternative_clinicaldata", "ML_apical", "models_apical")

    models_path = [path]

    # import ML models
    # descriptors = {}
    models = {}

    for model_path in models_path:
        files = os.listdir(model_path)
        print("models_apical")
        for file in files:
            if "pkl" in file:
                path = os.path.join(model_path, file)
                with open(path, 'rb') as file:
                    models["models_apical"] = pickle.load(file)
            """
            elif "train" in file:
                path = os.path.join(model_path, file)
                columns = pd.read_excel(path)
                descriptors["models_apical"] = list(columns.keys()[2:])
            """
    
    return models

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='cardiotoxicity assessment')

    # Add an argument for input
    parser.add_argument('--smiles', required=True, help='Specify target smiles')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_value = args.smiles

    # Your script logic here
    print(f"Input value: {input_value}")
    return input_value

def run_script_in_conda_env(smiles:str):
    # Combine activation, script execution, and deactivation in a single command
    # Specify the path to your Python script in the other environment
    """
    You must env the conda env defined by https://github.com/jrwnter/cddd
    """
    script_path = os.path.join(os.getcwd(), "cddd", "cddd_calculation.py")
    conda_env = 'cddd'

    conda_env_path = os.path.expanduser(f'~/miniconda3/envs/{conda_env}/bin/python')
    
    command = f"{conda_env_path} {script_path} --smiles {smiles}"
    # Run the entire command
    subprocess.run(command, shell=True, check=True)

def cddd_calculation(smiles:str):
    """
    You have to use subprocess to calcualte cddd descriptors
    in other environment.
    """
    # Run the script in the other virtual environment with the input argument
    try: 
        run_script_in_conda_env(smiles=smiles)
        path = os.path.join(os.getcwd(), "smiles_CDDD.csv")
        data = pd.read_csv(path)
        os.remove(path)
        return data
    except Exception as e:
        print(e)
        print("error in subprocess")

def transform_data(target_CDDD:pd.DataFrame, models:dict):
    """
    take as input the dictionary with descriptors for model.
    """ 
    # adjust colums name as str
    # transforma in str
    data = target_CDDD.loc[:, list(models['models_apical'].feature_names_in_)]

    return data

def inference(smiles:str, data:pd.DataFrame, models:dict):

    # Prediction
    pred = models['models_apical'].predict(data)
    results = {"models_apical": pred}

    results_final = {}
    for name, result in results.items():
        if result == 0: activity = 'Inactive'
        else: activity = 'Active'
        results_final[name] = activity   

    return pd.DataFrame(pd.DataFrame(results_final, index=[smiles]))

if __name__ == "__main__":

    # ask smiles of chemicals to evaluate
    smiles = main()
    # check smiles validity:
    check_smiles(smiles)
    # import models pipeline and descriptors
    print("[INFO]: Models import...")
    models = import_models()
    print("done")

    print("[INFO]: calculate CDDD descriptors for target...")
    target_CDDD = cddd_calculation(smiles)
    print("done")

    print("[INFO]: AD_evaluation...")
    ad_results = ad_evaluation(smiles)
    print("done")

    data = transform_data(target_CDDD=target_CDDD, models=models)
    
    results = inference(smiles=smiles, data=data, models=models)
    
    merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
    
    merged_df.to_csv('results.csv')

    print("done.. you can find the results in results.csv file")
    


    
    
   