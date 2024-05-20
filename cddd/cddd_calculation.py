import pandas as pd
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
import os
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Your script description here')

    # Add an argument for input
    parser.add_argument('--smiles', required=True, help='Specify input value')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_value = args.smiles

    # Your script logic here
    print(f"Input value: {input_value}")
    return input_value


if __name__ == '__main__':
    
    smiles = main()
    smiles_preprocess = preprocess_smiles(smiles)
    print(f"smiles preprocessed: {smiles_preprocess}")
    inference_model = InferenceModel()
    print('importend model')
    smiles_embedding_train = inference_model.seq_to_emb([smiles_preprocess])
    dataset = pd.DataFrame(smiles_embedding_train)
    dataset['SMILES'] = [smiles_preprocess] # per mantenere gli indici
    parent_directory = os.path.join(os.getcwd(), "smiles_CDDD.csv")
    dataset.to_csv(parent_directory)
    print("done...")
