from flask import Flask, jsonify, request, send_file
import pandas as pd
from PipelineAlternative_clinicaldata.ML_apical import inference as ML
from PipelineAlternative_clinicaldata.AI import inference as AI
from PipelineAlternative_clinicaldata.AOP_models import inference as AOP
from dotenv import load_dotenv
import os
import traceback
import tempfile

load_dotenv()

app = Flask(__name__)


@app.route("/ml/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    smiles = data.get("smiles")
    if not smiles:
        return jsonify({"error": "No SMILES input provided"}), 400

    try:
        ML.check_smiles(smiles)
        print("[INFO]: Models import...")
        models = ML.import_models()
        print("done")
        print("[INFO]: calculate CDDD descriptors for target...")
        target_CDDD = ML.cddd_calculation(smiles)
        print("done")
        print("[INFO]: AD_evaluation...")
        ad_results = ML.ad_evaluation(smiles)

        data = ML.transform_data(target_CDDD=target_CDDD, models=models)

        results = ML.inference(smiles=smiles, data=data, models=models)

        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
            merged_df.to_csv(tmpfile.name)
            temp_file_path = tmpfile.name

        print("done..")

        return send_file(temp_file_path, as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return str(e), 500


@app.route("/ai/evaluate", methods=["POST"])
def evaluate_ai():
    data = request.get_json()
    smiles = data.get("smiles")
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
        traceback.print_exc()
        return str(e), 500


@app.route("/aop/evaluate", methods=["POST"])
def evaluate_aop():
    data = request.get_json()
    smiles = data.get("smiles")
    if not smiles:
        return jsonify({"error": "No SMILES input provided"}), 400

    try:
        AOP.check_smiles(smiles)
        models = AOP.import_models()
        target_CDDD = AOP.cddd_calculation(smiles)
        ad_results = AOP.ad_evaluation(smiles)
        target_dict = AOP.transform_data(models=models, target_CDDD=target_CDDD)
        results = AOP.inference(models=models, smiles=smiles, target_dict=target_dict)
        
        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
            merged_df.to_csv(tmpfile.name)
            temp_file_path = tmpfile.name

        return send_file(temp_file_path, as_attachment=True)
    
    except Exception as e:
        traceback.print_exc()
        return str(e), 500

@app.route("/isalive", methods=["GET"])
def is_alive():
    return jsonify({"status": "alive"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
