from flask import Flask, jsonify, request, send_file
import pandas as pd
from PipelineAlternative_clinicaldata.ML_apical import inference as ML
from PipelineAlternative_clinicaldata.AI import inference as AI
from PipelineAlternative_clinicaldata.AOP_models import inference as AOP
from dotenv import load_dotenv
import os
import traceback
import io
import logging

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def validate_smiles(data):
    smiles = data.get("smiles")
    if not smiles:
        logger.warning("No SMILES input provided")
        return None, jsonify({"error": "No SMILES input provided"}), 400
    logger.debug(f"Received SMILES: {smiles}")
    return smiles, None, None

def generate_csv_response(merged_df):
    output = io.BytesIO()
    merged_df.to_csv(output, index=False)
    output.seek(0)
    logger.debug("Generated CSV response")
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name="results.csv")

@app.route("/clinicaldata/ml/evaluate", methods=["POST"])
def evaluate_ml():
    data = request.get_json()
    logger.debug("Received request for ML evaluation")
    smiles, error_response, status_code = validate_smiles(data)
    if error_response:
        return error_response, status_code

    try:
        ML.check_smiles(smiles)
        logger.debug("SMILES validation passed")
        models = ML.import_models()
        logger.debug("Models imported successfully")
        target_CDDD = ML.cddd_calculation(smiles)
        logger.debug("CDDD calculation completed")
        ad_results = ML.ad_evaluation(smiles, singlesmiles=True)
        logger.debug("AD evaluation completed")
        transformed_data = ML.transform_data(target_CDDD=target_CDDD, models=models)
        results = ML.inference(smiles=smiles, data=transformed_data, models=models)
        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
        return generate_csv_response(merged_df)
    except Exception as e:
        logger.error("Exception during ML evaluation", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/clinicaldata/ai/evaluate", methods=["POST"])
def evaluate_ai():
    data = request.get_json()
    logger.debug("Received request for AI evaluation")
    smiles, error_response, status_code = validate_smiles(data)
    if error_response:
        return error_response, status_code

    try:
        AI.check_smiles(smiles)
        logger.debug("SMILES validation passed")
        model_NLP = AI.import_NLP_model()
        logger.debug("NLP model imported successfully")
        target_NLP = AI.prepare_smiles_NLP(smiles)
        logger.debug("SMILES prepared for NLP")
        ad_results = AI.ad_evaluation(smiles)
        logger.debug("AD evaluation completed")
        inference_result = AI.inference_NLP(model_NLP, target_NLP)
        logger.debug("Inference completed")
        return jsonify({"prediction": inference_result})
    except Exception as e:
        logger.error("Exception during AI evaluation", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/clinicaldata/aop/evaluate", methods=["POST"])
def evaluate_aop():
    data = request.get_json()
    logger.debug("Received request for AOP evaluation")
    smiles, error_response, status_code = validate_smiles(data)
    if error_response:
        return error_response, status_code

    try:
        AOP.check_smiles(smiles)
        logger.debug("SMILES validation passed")
        models = AOP.import_models()
        logger.debug("Models imported successfully")
        target_CDDD = AOP.cddd_calculation(smiles)
        logger.debug("CDDD calculation completed")
        ad_results = AOP.ad_evaluation(smiles, singlesmiles=True)
        logger.debug("AD evaluation completed")
        target_dict = AOP.transform_data(models=models, target_CDDD=target_CDDD)
        results = AOP.inference(models=models, smiles=smiles, target_dict=target_dict)
        merged_df = pd.merge(results, ad_results, left_index=True, right_index=True)
        return generate_csv_response(merged_df)
    except Exception as e:
        logger.error("Exception during AOP evaluation", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/isalive", methods=["GET"])
def is_alive():
    logger.debug("Received isalive check")
    return jsonify({"status": "alive"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
