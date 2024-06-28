import contextvars
import io
import logging

from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from models.PipelineAlternative_clinicaldata.AI import inference as AI
from models.PipelineAlternative_clinicaldata.AOP_models import inference as AOP
from models.PipelineAlternative_clinicaldata.ML_apical import inference as ML

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# R context management
rpy2_context = contextvars.ContextVar('rpy2_context', default=None)

# Load the mrgsolve and httk packages in R
try:
    mrgsolve = importr('mrgsolve')
    httk = importr('httk')
except Exception as e:
    logger.error(f"Failed to import R packages: {e}")

@app.before_request
def before_request():
    # Set the context for rpy2 conversions
    rpy2_context.set(robjects.conversion.get_conversion())

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

def run_r_model(script_path, function_name, *args):
    try:
        robjects.r.source(script_path)
        r_func = robjects.globalenv[function_name]
        result = r_func(*args)
        return [list(row) for row in result]
    except Exception as e:
        logger.error(f"Error in R model simulation: {e}")
        return None

@app.route('/pbpk/doxorubicin', methods=['POST'])
def run_doxorubicin_model():
    data = request.json
    dose_mg = float(data.get('dose_mg', 60))
    age = int(data.get('age', 50))
    weight = float(data.get('weight', 70))
    height = float(data.get('height', 180))

    current_context = contextvars.copy_context()
    
    py_output = current_context.run(run_r_model, 'models/PBPK/dox_script_mrgsolve.R', 'run_doxorubicin_model', dose_mg, age, weight, height)
    
    if py_output is None:
        return jsonify({"error": "Failed to run doxorubicin model"}), 500
    
    return jsonify(py_output)

@app.route('/pbpk/httk', methods=['POST'])
def run_httk_model():
    data = request.json
    chem_name = data.get('chem_name', 'Bisphenol A')
    species = data.get('species', 'human')
    daily_dose = float(data.get('daily_dose', 1))
    doses_per_day = int(data.get('doses_per_day', 1))
    days = int(data.get('days', 15))

    current_context = contextvars.copy_context()
    
    py_output = current_context.run(run_r_model, 'models/PBPK/httk/scripts/setup.r', 'run_httk_model', chem_name, species, daily_dose, doses_per_day, days)
    
    if py_output is None:
        return jsonify({"error": "Failed to run httk model"}), 500
    
    return jsonify(py_output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
