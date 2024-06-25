import pandas as pd

import pickle
import os
import argparse
import subprocess
import sys

# tensorflow
import tensorflow as tf
# deepchem
import deepchem as dc

# rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

# from rdkit import Chem
from deepchem.feat import MordredDescriptors

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='[KE2]: Mitochondrial Toxicity assessment for cardiotoxicity using AI methods')

    # Add an argument for input
    parser.add_argument('--smiles', required=True, help='Specify target smiles')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_value = args.smiles

    # Your script logic here
    print(f"Input value: {input_value}")
    return input_value

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
    path_folder = os.path.join(os.getcwd(), "PipelineAlternative_clinicaldata", "AI", "data", "data.csv")
    data = pd.read_csv(path_folder)
    smiles_list = list(data['SMILES'])
    counter = 0
    for smi in smiles_list:
        fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=num_bits)
        if counter < 3: 
            if DataStructs.DiceSimilarity(fp1, fp2) > 0.2: 
                counter += 1     
        if counter < 3: ad_results["Applicability Domain"] = 'out AD'
        else: ad_results["Applicability Domain"] = 'in AD'

    return pd.DataFrame(ad_results, index=[smiles])

def import_NLP_model():
    "import models find in"
    path = os.path.join(os.getcwd(), "PipelineAlternative_clinicaldata", "AI", "charsEmbeddingAugemted_90_10")
    path_model = os.path.join(path, "Model_charsEmbeddingAugemted")
    return tf.keras.models.load_model(path_model)

def run_script_in_conda_env(smiles:str):
    # Combine activation, script execution, and deactivation in a single command
    # Specify the path to your Python script in the other environment
    """
    You must env the conda env defined by https://github.com/jrwnter/cddd
    """
    script_path = os.path.join(os.getcwd(), "cddd", "cddd_calculation.py")
    conda_env = 'cddd'
    
    conda_env_path = os.path.expanduser(f'~/miniconda3/envs/{conda_env}/bin/python')
    
    command = f"{conda_env_path} {script_path} --smiles \"{smiles}\""
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
        return tf.expand_dims(tf.constant(data.values()))
    except: 
        print("error in subprocess")

def mordred_descriptors(smiles:str):
    # descriptors calculators mordred
    featurizer_descriptors = MordredDescriptors(ignore_3D=True)
    columns_descriptors = [f"descriptors_{i}" for i in range(1613)]
    # descriptors calculator for training
    features_descriptors = featurizer_descriptors.featurize(smiles)
    # target = pd.DataFrame(features_descriptors, columns=columns_descriptors)
    with open("my_pipeline.pkl", "rb") as f: pipeline=pickle.load(f)
    # target = pd.DataFrame(, columns=columns_descriptors))
    return tf.expand_dims(tf.constant(pipeline.transform(features_descriptors)))

def fingerprint_cfp(smiles:str):
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=4)    
    return tf.expand_dims(tf.constant([int(i) for i in featurizer.featurize(smiles).tolist()[0]]))

def fingerprint_MACCs(smiles):
    featurizer = dc.feat.MACCSKeysFingerprint()
    return tf.expand_dims(tf.constant(featurizer.featurize(smiles).tolist()[0]))

def prepare_smiles_NLP(text): 
    smi = " ".join(list(text))
    return tf.expand_dims(tf.constant(smi, dtype=tf.string), axis=0)

def inference_multimodal(model, target_NLP, target_CDDD, target_MD, target_MACCs, target_cfp):
    input_data = (target_NLP, target_cfp, target_MACCs, target_CDDD, target_MD) # problema.. sembrerebbe che vadano selezionati i descrittori MD non solo scalati
    pass

def inference_NLP(model, target_NLP):
    pred = (model.predict(target_NLP).squeeze() > 0.5).astype(int)
    return "Inactive" if pred == 0 else "Active"

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
    model_NLP = import_NLP_model()
    print("done")
    """
    print("[INFO]: calculate CDDD descriptors for target...")
    target_CDDD = cddd_calculation(smiles)
    print("done")

    print("[INFO]: calculate MD descriptors for target...")
    target_MD = mordred_descriptors(smiles)
    print("done")

    print("[INFO]: calculate MACCs descriptors for target...")
    target_MACCs = fingerprint_MACCs(smiles)
    print("done")

    print("[INFO]: calculate CFP descriptors for target...")
    target_cfp = fingerprint_cfp(smiles)
    print("done")
    """
    print("[INFO]: calculate SMILES character embedding for target...")
    target_NLP = prepare_smiles_NLP(smiles)
    print("done")

    print("[INFO]: AD_evaluation...")
    ad_results = ad_evaluation(smiles)
    print("done")

    print("[INFO]: NLP inference...")
    inference = inference_NLP(model_NLP, target_NLP)
    print(f"[END ASSESSMENT] Prediction for {smiles}: \n\n\t{inference}")
    


    
    
   