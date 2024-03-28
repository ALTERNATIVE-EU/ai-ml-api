from flask import Flask, jsonify, request, send_file
import pandas as pd
from keycloak import KeycloakOpenID
from ML import inference as ML
from AI import inference as AI
from dotenv import load_dotenv
import os

load_dotenv()

# Keycloak Config
KEYCLOAK_URL = os.getenv('KEYCLOAK_URL')
KEYCLOAK_REALM = os.getenv('KEYCLOAK_REALM')
KEYCLOAK_CLIENT_ID = os.getenv('KEYCLOAK_CLIENT_ID')
KEYCLOAK_CLIENT_SECRET = os.getenv('KEYCLOAK_CLIENT_SECRET')

# Keycloak client
keycloak_openid = KeycloakOpenID(server_url=KEYCLOAK_URL,
                                 client_id=KEYCLOAK_CLIENT_ID,
                                 realm_name=KEYCLOAK_REALM,
                                 client_secret_key=KEYCLOAK_CLIENT_SECRET)

app = Flask(__name__)

@app.route('/ml/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({"error": "No SMILES input provided"}), 400

    try:
        ML.check_smiles(smiles)
        models, descriptors, pipeline = ML.import_models()
        target_CDDD = ML.cddd_calculation(smiles)
        target_MD = ML.mordred_descriptors(smiles)
        ad_results = ML.ad_evaluation(smiles)
        data_mie, data_ke1, data_ke2 = ML.transform_data(target_MD, target_CDDD, descriptors, pipeline, models)
        results = ML.inference(smiles, data_mie, data_ke1, data_ke2, models)
        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
        merged_df.to_csv('results.csv')

        return send_file('results.csv', as_attachment=True)
    except Exception as e:
        return str(e), 500

@app.route('/ai/evaluate', methods=['POST'])
def evaluate_ai():
    data = request.get_json()
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({"error": "No SMILES input provided"}), 400

    try:
        AI.check_smiles(smiles)
        model_NLP = AI.import_NLP_model()
        target_NLP = AI.prepare_smiles_NLP(smiles)
        ad_results = AI.ad_evaluation(smiles)
        inference = AI.inference_NLP(model_NLP, target_NLP)
        return jsonify({"prediction": inference})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)