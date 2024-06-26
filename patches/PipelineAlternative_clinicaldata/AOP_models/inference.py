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

def ad_evaluation(smiles:str, radius=3, num_bits=1024, singlesmiles=False):
    """
    check similarity between compounds in dataset and target using morgan fingerprint as descriptors
    """
    fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits=num_bits)
    ad_results = {}
    path_folder = os.path.join(os.getcwd(), "models", "PipelineAlternative_clinicaldata", "AOP_models", "data")
    for file in os.listdir(path_folder):
        print(file)
        path_folder_endpoint = os.path.join(path_folder, file)
        name = file.split("_")[0]
        data = pd.read_csv(path_folder_endpoint)
        smiles_list = list(data['smiles'])
        counter = 0
        for smi in smiles_list:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=num_bits)
            if counter < 3: 
                if DataStructs.DiceSimilarity(fp1, fp2) > 0.3: 
                    counter += 1
        
        if counter < 3: ad_results[f"AD_{name}"] = 'out AD'
        else: ad_results[f"AD_{name}"] = 'in AD'

    if singlesmiles: return pd.DataFrame([ad_results], index=[smiles])
    else: return pd.DataFrame([ad_results])

def import_models():
    "import models find in BestModels_ML_90_10 folder"

    path = os.path.join(os.getcwd(), "models", "PipelineAlternative_clinicaldata", "AOP_models", "models")
    models_path = {file.split("_")[0]:os.path.join(path, file) for file in os.listdir(path)}

    # import ML models
    models = {}

    for key, model_path in models_path.items():
        print(key)
        with open(model_path, 'rb') as file:
            models[key] = pickle.load(file)
    return models

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='cardiotoxicity assessment')

    # Add an argument for input
    parser.add_argument('--smiles', required=False, help='Specify target smiles')
    parser.add_argument('--filename', required=False, help='Specify csv files with "smiles" columns')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_value = args.smiles
    if input_value is None:
        input_value = args.filename

    # Your script logic here
    print(f"Input value: {input_value}")
    return input_value

def run_script_in_conda_env(smiles:str, singlesmiles=True):
    # Combine activation, script execution, and deactivation in a single command
    # Specify the path to your Python script in the other environment
    """
    You must env the conda env defined by https://github.com/jrwnter/cddd
    """
    script_path = os.path.join(os.getcwd(), "models", "PipelineAlternative_clinicaldata", "cddd", "cddd_calculation.py")
    conda_env = 'cddd'
    
    conda_env_path = os.path.expanduser(f'~/miniconda3/envs/{conda_env}/bin/python')

    if singlesmiles: 
        print("SINGLE MOLECULE")
        command = f"{conda_env_path} {script_path} --smiles \"{smiles}\""
    else:  
        print("DATASET OF MOLECULEs")
        command = f"{conda_env_path} {script_path} --file {smiles}"
    # cddd --input smiles.smi --output descriptors.csv  --smiles_header smiles
    # Run the entire command
    subprocess.run(command, shell=True, check=True)

def cddd_calculation(smiles:str, singlesmiles=True):
    """
    You have to use subprocess to calcualte cddd descriptors
    in other environment.
    """
    # Run the script in the other virtual environment with the input argument
    try: 
        run_script_in_conda_env(smiles=smiles, singlesmiles=singlesmiles)
        path = os.path.join(os.getcwd(), "smiles_CDDD.csv")
        data = pd.read_csv(path)
        os.remove(path)
        return data
    except Exception as e:
        print("Error in cddd calculation")
        print(e)

def transform_data(models:dict, target_CDDD:pd.DataFrame):
    """
    take as input the dictionary with descriptors for each model
    and select only the useful ones.
    """ 
    # adjust colums name as str
    # transforma in str
    target_dict = {}
    for key, value in models.items():
        columns = list(value.feature_names_in_)
        target_dict[key] = target_CDDD.loc[:, columns]
    
    return target_dict

def inference(models:dict, smiles, target_dict:dict, singlemol=True):

    # Prediction
    results_final = {}
    for key, model in models.items():
        results_final[key] = model.predict(target_dict[key])

    if singlemol: return pd.DataFrame(pd.DataFrame(results_final, index=[smiles]))
    else: return pd.DataFrame(pd.DataFrame([results_final]))#, index=smiles)

if __name__ == "__main__":

    # ask smiles of chemicals to evaluate
    smiles = main()
    if ".csv" in smiles:
        print("file to manage:")
        data = pd.read_csv(smiles)
        for smi in data['smiles']:
            check_smiles(smi)

        # import models pipeline and descriptors
        print("[INFO]: Models import...")
        models = import_models()
        print("done")

        print("[INFO]: calculate CDDD descriptors for target...")
        target_CDDD = cddd_calculation(smiles, singlesmiles=False)
        print(f"{target_CDDD}done")

        print("[INFO]: AD_evaluation...")
        for n, smi in enumerate(target_CDDD['SMILES']):
            ad = ad_evaluation(smi)
            if n == 0: ad_results = ad.copy()
            else: 
                ad_results = pd.concat([ad_results, ad], axis=0)
        # debug
        ad_results.reset_index(inplace=True, drop=True)
        print("done")

        target_dict = transform_data(models=models, target_CDDD=target_CDDD)
        results = inference(models=models,
                            smiles=target_CDDD['SMILES'].tolist(), 
                            target_dict=target_dict)
        
        results.reset_index(drop=True, inplace=True)
        results['smiles'] = target_CDDD['SMILES'].tolist()
        merged_df = pd.concat([results, ad_results], axis=1)
        merged_df.to_csv('results.csv')
        print("done.. you can find the results in results.csv file")

    else:
        # check smiles validity:
        check_smiles(smiles)
        # import models pipeline and descriptors
        print("[INFO]: Models import...")
        models = import_models()
        print("done")

        print("[INFO]: calculate CDDD descriptors for target...")
        target_CDDD = cddd_calculation(smiles)
        print("done")
        print(target_CDDD)

        print("[INFO]: AD_evaluation...")
        ad_results = ad_evaluation(smiles, singlesmiles=True)
        print("done")
        print(ad_results)

        target_dict = transform_data(models=models, target_CDDD=target_CDDD)
        
        results = inference(models=models,
                            smiles=smiles, 
                            target_dict=target_dict)
        print(results)
        
        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
        
        merged_df.to_csv('results.csv')

        print("done.. you can find the results in results.csv file")
    


    
    
   