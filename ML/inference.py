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

# from rdkit import Chem
from deepchem.feat import MordredDescriptors

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    path_folder = os.path.join(FILE_DIR, "data")
    for folder in os.listdir(path_folder):
        print(folder)
        path_folder_endpoint = os.path.join(path_folder, folder, f'FP_data.csv')
        name = folder.split("_")[0]
        data = pd.read_csv(path_folder_endpoint)
        smiles_list = list(data['SMILES'])
        counter = 0
        for smi in smiles_list:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=num_bits)
            if counter < 3: 
                if DataStructs.DiceSimilarity(fp1, fp2) > 0.2: 
                    counter += 1
        
        if counter < 3: ad_results[folder] = 'out AD'
        else: ad_results[f"{name}_AD"] = 'in AD'

    return pd.DataFrame(ad_results, index=[smiles])

def import_models():
    "import models find in BestModels_ML_90_10 folder"

    path = os.path.join(FILE_DIR, "BestModels_ML_90_10")

    path_mie = os.path.join(path, "MIE")
    path_ke1 = os.path.join(path, "KE1")
    path_ke2 = os.path.join(path, "KE2")

    models_path = [path_mie, path_ke1, path_ke2]

    # import ML models
    descriptors = {}
    models = {}
    pipeline = {}

    for model_path in models_path:
        files = os.listdir(model_path)
        name = model_path[-3:]
        print(name)
        for file in files:
            if "pipeline" in file:
                path = os.path.join(model_path, file)
                with open(path, 'rb') as file:
                    pipeline[name] = pickle.load(file)
            elif "MD" in file or "CDDD" in file:
                path = os.path.join(model_path, file)
                with open(path, 'rb') as file:
                    models[name] = pickle.load(file)
            elif "descriptors" in file:
                path = os.path.join(model_path, file)
                with open(path, 'rb') as file:
                    descriptors[name] = pickle.load(file)
    
    return models, descriptors, pipeline

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
    script_path = os.path.join(FILE_DIR, "cddd", "cddd_calculation.py")
    conda_env = 'cddd'
    conda_init_script = '~/miniconda3/etc/profile.d/conda.sh'  # Adjust based on your Conda installation
    command = f". {conda_init_script} && conda activate {conda_env} && python {script_path} --smiles {smiles} && conda deactivate"
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

def mordred_descriptors(smiles:str):
    # descriptors calculators mordred
    featurizer_descriptors = MordredDescriptors(ignore_3D=True)
    columns_descriptors = [f"descriptors_{i}" for i in range(1613)]
    # descriptors calculator for training
    features_descriptors = featurizer_descriptors.featurize(smiles)
    target = pd.DataFrame(features_descriptors, columns=columns_descriptors)
    return target

def transform_data(target_MD:pd.DataFrame,
                   target_CDDD:pd.DataFrame,
                   descriptors:dict,
                    pipeline:dict,
                    models:dict):
    """
    take as input the dictionary with descriptors for each model
    and select only the useful ones.
    """ 
    # adjust colums name as str
    # transforma in str
    descriptors['KE2'] = [str(i) for i in descriptors['KE2']]
    # select only specific columns
    target_dict = {}
    for key, value in descriptors.items():
        if key != "KE2":
            target_dict[key] = target_MD.loc[:, value]

    # transforma in str
    descriptors['KE2'] = [str(i) for i in descriptors['KE2']]

    # seleziona solo le colonne utili
    target_dict = {}
    for key, value in descriptors.items():
        if key != "KE2":
            target_dict[key] = target_MD.loc[:, value]
            
    data_mie = pd.DataFrame(pipeline['MIE'].transform(target_dict['MIE']), columns=descriptors["MIE"])
    data_ke1 = pd.DataFrame(pipeline['KE1'].transform(target_dict['KE1']), columns=descriptors["KE1"])


    data_mie = data_mie.loc[:, models['MIE'].feature_names_in_]
    data_ke1 = data_ke1.loc[:, models['KE1'].feature_names_in_]
    data_ke2 = target_CDDD.loc[:, list(models['KE2'].feature_names_in_)]

    return data_mie, data_ke1, data_ke2

def inference(smiles:str, data_mie:pd.DataFrame, data_ke1:pd.DataFrame, data_ke2:pd.DataFrame, models:dict):

    # Prediction
    pred_mie = models['MIE'].predict(data_mie)
    pred_ke1 = models['KE1'].predict(data_ke1)
    pred_ke2 = models['KE2'].predict(data_ke2)

    results = {"MIE": pred_mie, "KE1": pred_ke1, "KE2": pred_ke2}

    results_final = {}
    for name, result in results.items():
        if result == 0: activity = 'Inactive'
        else: activity = 'Active'
        results_final[name] = activity   

    return pd.DataFrame(pd.DataFrame(results_final, index=[smiles]))

"""
def results_printed(merged_df:pd.DataFrame):
     if name == "MIE":
            print(f"The compounds {smiles} is tested for inhibition of mitochondrial complexes (MIE1):\n\n\tEvaluation: {activity}\n\n\n")
        elif name == "KE1":
            print(f"The compounds {smiles} is tested for Increase of oxidative stress (KE1):\n\n\tEvaluation: {activity}\n\n\n")
        elif name == "KE2":
                print(f"The compounds {smiles} is tested for Increase Mitochondrial Dysfunction (KE2):\n\n\tEvaluation: {activity}")
"""
if __name__ == "__main__":

    # ask smiles of chemicals to evaluate
    smiles = main()
    # check smiles validity:
    check_smiles(smiles)
    # import models pipeline and descriptors
    print("[INFO]: Models import...")
    models, descriptors, pipeline = import_models()
    print("done")

    print("[INFO]: calculate CDDD descriptors for target...")
    target_CDDD = cddd_calculation(smiles)
    print("done")

    print("[INFO]: calculate MD descriptors for target...")
    target_MD = mordred_descriptors(smiles)
    print("done")

    print("[INFO]: AD_evaluation...")
    ad_results = ad_evaluation(smiles)
    print("done")

    data_mie, data_ke1, data_ke2 = transform_data(target_MD=target_MD,
                                                target_CDDD=target_CDDD,
                                                descriptors=descriptors,
                                                pipeline=pipeline,
                                                models=models)
    results = inference(smiles=smiles, 
                        data_mie=data_mie, 
                        data_ke1=data_ke1, 
                        data_ke2=data_ke2,
                        models=models)
    

    merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
    
    merged_df.to_csv('results.csv')

    print("done.. you can find the results in results.csv file")
    


    
    
   