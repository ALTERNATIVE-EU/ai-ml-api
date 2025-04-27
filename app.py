from flask import Flask, request, send_file, jsonify
from flask_restx import Api, Resource, fields
from dotenv import load_dotenv
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import contextvars
import io
import logging
from werkzeug.exceptions import BadRequest
import torch

# Import your model modules here
from models.PipelineAlternative_clinicaldata.AI import inference as AI
from models.PipelineAlternative_clinicaldata.AOP_models import inference as AOP
from models.PipelineAlternative_clinicaldata.ML_apical import inference as ML
from models.PipelineAlternative_clinicaldata.ModelhERG import Inference as hERG
from models.PipelineAlternative_clinicaldata.Multitask_deploy import inference as Multitask
from models.PipelineAlternative_clinicaldata.AHRModel import inference as AHR

load_dotenv()

# Define authorization model for Swagger
authorizations = {
    'Bearer': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': "Type in the *'Value'* input box below: **'Bearer &lt;JWT&gt;'**, where JWT is the token"
    }
}

app = Flask(__name__)
api = Api(app, 
          version='1.0', 
          title='Clinical Data API',
          description='A suite of clinical data evaluation endpoints',
          doc='/swagger', 
          authorizations=authorizations,
          security='Bearer'
)

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

# Define namespaces
ns_clinical = api.namespace('clinicaldata', description='Clinical data operations')
ns_pbpk = api.namespace('pbpk', description='PBPK model operations')

# Define models
smiles_model = api.model('SMILES', {
    'smiles': fields.String(required=True, description='SMILES string of the compound')
})

doxorubicin_model = api.model('Doxorubicin', {
    'dose_mg': fields.Float(default=60, description='Dose in mg'),
    'age': fields.Integer(default=50, description='Age in years'),
    'weight': fields.Float(default=70, description='Weight in kg'),
    'height': fields.Float(default=190, description='Height in cm')
})

httk_model = api.model('HTTK', {
    'chem_name': fields.String(default='Bisphenol A', description='Chemical name'),
    'species': fields.String(default='human', description='Species'),
    'daily_dose': fields.Float(default=1, description='Daily dose'),
    'doses_per_day': fields.Integer(default=1, description='Doses per day'),
    'days': fields.Integer(default=15, description='Number of days')
})

def validate_smiles(data):
    smiles = data.get("smiles")
    if not smiles:
        logger.warning("No SMILES input provided")
        raise BadRequest("No SMILES input provided")
    if not isinstance(smiles, str):
        logger.warning("SMILES input must be a string")
        raise BadRequest("SMILES input must be a string")
    logger.debug(f"Received SMILES: {smiles}")
    return smiles

def validate_doxorubicin_params(data):
    dose_mg = data.get('dose_mg', 60)
    age = data.get('age', 50)
    weight = data.get('weight', 70)
    height = data.get('height', 190)

    if not isinstance(dose_mg, (int, float)) or dose_mg <= 0:
        raise BadRequest("dose_mg must be a positive number")
    if not isinstance(age, int) or age <= 0:
        raise BadRequest("age must be a positive integer")
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise BadRequest("weight must be a positive number")
    if not isinstance(height, (int, float)) or height <= 0:
        raise BadRequest("height must be a positive number")

    return float(dose_mg), int(age), float(weight), float(height)

def validate_httk_params(data):
    chem_name = data.get('chem_name', 'Bisphenol A')
    species = data.get('species', 'human')
    daily_dose = data.get('daily_dose', 1)
    doses_per_day = data.get('doses_per_day', 1)
    days = data.get('days', 15)

    if not isinstance(chem_name, str):
        raise BadRequest("chem_name must be a string")
    if not isinstance(species, str):
        raise BadRequest("species must be a string")
    if not isinstance(daily_dose, (int, float)) or daily_dose <= 0:
        raise BadRequest("daily_dose must be a positive number")
    if not isinstance(doses_per_day, int) or doses_per_day <= 0:
        raise BadRequest("doses_per_day must be a positive integer")
    if not isinstance(days, int) or days <= 0:
        raise BadRequest("days must be a positive integer")

    return chem_name, species, float(daily_dose), int(doses_per_day), int(days)

def generate_csv_response(merged_df):
    output = io.BytesIO()
    merged_df.to_csv(output, index=False)
    output.seek(0)
    logger.debug("Generated CSV response")
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name="results.csv")

@ns_clinical.route('/herg/evaluate')
class HERGEvaluate(Resource):
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='Prediction result'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            smiles = validate_smiles(data)
            
            hERG.check_smiles(smiles)
            logger.debug("SMILES validation passed")
            model_hERG = hERG.import_models()
            logger.debug("hERG model imported successfully")
            target_CDDD = hERG.cddd_calculation(smiles, singlesmiles=True)
            logger.debug("CDDD calculation completed")
            data_test = target_CDDD.loc[:, model_hERG.feature_names_in_]
            prediction = model_hERG.predict(data_test)
            logger.debug(f"Prediction completed: {prediction!r}")
            print("--- Prediction:", prediction[0])
            prediction_list = prediction.tolist()  # Convert ndarray to Python list
            data['hERG channel blockade activity predicted'] = prediction_list
            data['applicability domain'] = hERG.localOutlierFactor_applicability_domain(data_test)
            logger.debug("Prediction completed")
            return {"prediction": prediction_list[0], "applicability_domain": data['applicability domain'][0]}
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during hERG evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@ns_clinical.route('/ahr/evaluate')
class MultitaskEvaluate(Resource):
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='CSV file with results'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            smiles = validate_smiles(data)
            d = {'SMILES': [smiles]}
            df = pd.DataFrame(data=d)

            data = df
            data_md_all = AHR.mordred_calculator(data)
            print("Mordred descriptors calculated")
            model, pipeline, train = AHR.import_model_pipeline()
            print("Model and pipeline loaded")
            numerical_cols = pipeline.feature_names_in_.tolist()
            columns_for_model = model.feature_names_in_.tolist()
            data_md_scaled = pd.DataFrame(pipeline.transform(data_md_all.loc[:, numerical_cols]), columns = numerical_cols).loc[:, columns_for_model]
            
            data['AHR_assesment'] = model.predict(data_md_scaled)
            data['ApplicabilityDomain'] = AHR.localOutlierFactor_applicability_domain(train, data_md_scaled)
            
            # Return result as JSON
            print(data.to_json(orient="records"))
            
            # Return results as JSON
            results_json = data.to_dict(orient='records')
            return jsonify(results_json[0])
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during multitask evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500
@ns_clinical.route('/multitask/evaluate')
class MultitaskEvaluate(Resource):
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='CSV file with results'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            tasks_names = ['Apical cardiotoxicity',
                    'Aryl hydrocarbon receptor',
                    'Cardiomyocyte Myocardial Injury',
                    'Change Action Potential',
                    'Change in Inotropy',
                    'Change In Vasoactivity',
                    'Endothelial injury coagulation',
                    'hERG channels inhibitors',
                    'Increase mitochondrial dysfunction',
                    'Inhibition mitochondrial complexes',
                    'OxidativeStress',
                    'Valvular Injury Proliferation']
            
            data = request.json
            smiles = validate_smiles(data)
            
            Multitask.check_smiles(smiles)
            logger.debug("SMILES validation passed")
            model, thr = Multitask.import_models()
            logger.debug("Multitask model imported successfully")

            target_CDDD = Multitask.cddd_calculation(smiles, singlesmiles=True)
            print(f"{target_CDDD}done")

            print("Model type:", type(model))
            print("target_CDDD.columns: ", target_CDDD.columns)
            data_test = torch.tensor(target_CDDD.drop(['original_smiles', 'SMILES'], axis=1).values, dtype=torch.float32)
            x = {'CDDD': data_test}
            prediction = model(x)
            if prediction.shape[0] == 1:
                results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int)).T
                results.columns = tasks_names
            else:
                results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int), columns=tasks_names)

            results_ = pd.concat([target_CDDD.loc[:, ['original_smiles', 'SMILES']], results], axis=1)
            results['ApplicabilityDomain'] = Multitask.localOutlierFactor_applicability_domain(target_CDDD.drop(['original_smiles', 'SMILES'], axis=1).values)
            
            # remove prediction for the task that don't reach satisfactory results
            results.drop(['Inhibition mitochondrial complexes'], axis=1, inplace=True)
            
            # Return results as JSON
            results_json = results.to_dict(orient='records')
            return jsonify(results_json[0])
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during multitask evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@ns_clinical.route('/ml/evaluate')
class MLEvaluate(Resource):
    @api.doc(security='Bearer')
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='CSV file with results'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            smiles = validate_smiles(data)
            
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
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during ML evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@ns_clinical.route('/ai/evaluate')
class AIEvaluate(Resource):
    @api.doc(security='Bearer')
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='Prediction result'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            smiles = validate_smiles(data)
            
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
            return {"prediction": inference_result}
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during AI evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@ns_clinical.route('/aop/evaluate')
class AOPEvaluate(Resource):
    @api.doc(security='Bearer')
    @api.expect(smiles_model)
    @api.response(200, 'Success', fields.String(description='CSV file with results'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            smiles = validate_smiles(data)
            
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
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during AOP evaluation: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

def run_r_model(script_path, function_name, *args):
    try:
        robjects.r.source(script_path)
        r_func = robjects.globalenv[function_name]
        result = r_func(*args)
        return [list(row) for row in result]
    except Exception as e:
        logger.error(f"Error in R model simulation: {e}")
        return None

@ns_pbpk.route('/doxorubicin')
class DoxorubicinModel(Resource):
    @api.doc(security='Bearer')
    @api.expect(doxorubicin_model)
    @api.response(200, 'Success', fields.String(description='Model output'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            dose_mg, age, weight, height = validate_doxorubicin_params(data)

            current_context = contextvars.copy_context()
            
            py_output = current_context.run(run_r_model, 'models/PBPK/dox_script_mrgsolve.R', 'run_doxorubicin_model', dose_mg, age, weight, height)
            
            if py_output is None:
                return {"error": "Failed to run doxorubicin model"}, 500
            
            return jsonify(py_output)
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during doxorubicin model: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@ns_pbpk.route('/httk')
class HTTKModel(Resource):
    @api.doc(security='Bearer')
    @api.expect(httk_model)
    @api.response(200, 'Success', fields.String(description='Model output'))
    @api.response(400, 'Bad Request', fields.String(description='Error message'))
    @api.response(500, 'Internal Server Error', fields.String(description='Error message'))
    def post(self):
        try:
            data = request.json
            chem_name, species, daily_dose, doses_per_day, days = validate_httk_params(data)

            current_context = contextvars.copy_context()
            
            py_output = current_context.run(run_r_model, 'models/PBPK/httk/scripts/setup.r', 'run_httk_model', chem_name, species, daily_dose, doses_per_day, days)
            
            if py_output is None:
                return {"error": "Failed to run httk model"}, 500
            
            return jsonify(py_output)
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Exception during httk model: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

@api.route('/isalive')
class IsAlive(Resource):
    @api.doc(security='Bearer')
    def get(self):
        logger.debug("Received isalive check")
        return {"status": "alive"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
