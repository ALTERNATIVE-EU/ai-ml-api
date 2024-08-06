import unittest
from flask import Flask, jsonify, request, send_file
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd
from app import run_r_model
import json
import os

from app import app


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # /ml/evaluate tests
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.check_smiles")
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.import_models")
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.cddd_calculation")
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.ad_evaluation")
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.transform_data")
    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.inference")
    @patch("pandas.merge")
    def test_ml_evaluate_valid(
        self,
        mock_merge,
        mock_inference,
        mock_transform_data,
        mock_ad_evaluation,
        mock_cddd_calculation,
        mock_import_models,
        mock_check_smiles,
    ):
        mock_check_smiles.return_value = None
        mock_import_models.return_value = "models"
        mock_cddd_calculation.return_value = "cddd"
        mock_ad_evaluation.return_value = pd.DataFrame()
        mock_transform_data.return_value = "data"
        mock_inference.return_value = pd.DataFrame()
        mock_merge.return_value = pd.DataFrame()

        response = self.app.post("/clinicaldata/ml/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Content-Disposition"], "attachment; filename=results.csv"
        )

    def test_ml_evaluate_no_smiles(self):
        response = self.app.post("/clinicaldata/ml/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    def test_ml_evaluate_empty_smiles(self):
        response = self.app.post("/clinicaldata/ml/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    @patch("models.PipelineAlternative_clinicaldata.ML_apical.inference.check_smiles")
    def test_ml_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/clinicaldata/ml/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Internal server error", response.get_data(as_text=True))

    # /ai/evaluate tests
    @patch("models.PipelineAlternative_clinicaldata.AI.inference.check_smiles")
    @patch("models.PipelineAlternative_clinicaldata.AI.inference.import_NLP_model")
    @patch("models.PipelineAlternative_clinicaldata.AI.inference.prepare_smiles_NLP")
    @patch("models.PipelineAlternative_clinicaldata.AI.inference.ad_evaluation")
    @patch("models.PipelineAlternative_clinicaldata.AI.inference.inference_NLP")
    def test_ai_evaluate_valid(
        self,
        mock_inference_NLP,
        mock_ad_evaluation,
        mock_prepare_smiles_NLP,
        mock_import_NLP_model,
        mock_check_smiles,
    ):
        mock_check_smiles.return_value = None
        mock_import_NLP_model.return_value = "model_NLP"
        mock_prepare_smiles_NLP.return_value = "target_NLP"
        mock_ad_evaluation.return_value = pd.DataFrame()
        mock_inference_NLP.return_value = "inference"

        response = self.app.post("/clinicaldata/ai/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"prediction": "inference"})

    def test_ai_evaluate_no_smiles(self):
        response = self.app.post("/clinicaldata/ai/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    def test_ai_evaluate_empty_smiles(self):
        response = self.app.post("/clinicaldata/ai/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    @patch("models.PipelineAlternative_clinicaldata.AI.inference.check_smiles")
    def test_ai_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/clinicaldata/ai/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Internal server error", response.get_data(as_text=True))


    # /aop/evaluate tests
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.check_smiles")
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.import_models")
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.cddd_calculation")
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.ad_evaluation")
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.transform_data")
    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.inference")
    @patch("pandas.merge")
    def test_aop_evaluate_valid(
        self,
        mock_merge,
        mock_inference,
        mock_transform_data,
        mock_ad_evaluation,
        mock_cddd_calculation,
        mock_import_models,
        mock_check_smiles,
    ):
        mock_check_smiles.return_value = None
        mock_import_models.return_value = "models"
        mock_cddd_calculation.return_value = "cddd"
        mock_ad_evaluation.return_value = pd.DataFrame()
        mock_transform_data.return_value = "data"
        mock_inference.return_value = pd.DataFrame()
        mock_merge.return_value = pd.DataFrame()

        response = self.app.post("/clinicaldata/aop/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Content-Disposition"], "attachment; filename=results.csv"
        )

    def test_aop_evaluate_no_smiles(self):
        response = self.app.post("/clinicaldata/aop/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    def test_aop_evaluate_empty_smiles(self):
        response = self.app.post("/clinicaldata/aop/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: No SMILES input provided"})

    @patch("models.PipelineAlternative_clinicaldata.AOP_models.inference.check_smiles")
    def test_aop_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/clinicaldata/aop/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Internal server error", response.get_data(as_text=True))

    # /isalive tests
    def test_isalive(self):
        response = self.app.get("/isalive")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "alive"})


    # /pbpk/doxorubicin tests
    @patch("app.run_r_model")
    def test_doxorubicin_valid(self, mock_run_r_model):
        mock_run_r_model.return_value = "output"

        response = self.app.post("/pbpk/doxorubicin", json={"dose_mg": 60, "age": 50, "weight": 70, "height": 190})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, "output")
        
    @patch("app.run_r_model")
    def test_doxorubicin_no_params(self, mock_run_r_model):
        mock_run_r_model.return_value = "output"        
        response = self.app.post("/pbpk/doxorubicin", json={})
        
        response = self.app.post("/pbpk/doxorubicin", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, "output")
        
    def test_doxorubicin_invalid_params(self):
        response = self.app.post("/pbpk/doxorubicin", json={"dose_mg": -1, "age": 50, "weight": 70, "height": 190})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: dose_mg must be a positive number"})
        
    @patch("app.run_r_model")
    def test_doxorubicin_run_r_model_exception(self, mock_run_r_model):
        mock_run_r_model.side_effect = Exception("Run R Model Error")
        response = self.app.post("/pbpk/doxorubicin", json={"dose_mg": 60, "age": 50, "weight": 70, "height": 190})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Internal server error", response.get_data(as_text=True))
        
    # /pbpk/httk tests
    @patch("app.run_r_model")
    def test_httk_valid(self, mock_run_r_model):
        mock_run_r_model.return_value = "output"

        response = self.app.post("/pbpk/httk", json={"chem_name": "Bisphenol A", "species": "human", "daily_dose": 1, "doses_per_day": 1, "days": 15})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, "output")

    @patch("app.run_r_model")
    def test_httk_no_params(self, mock_run_r_model):
        mock_run_r_model.return_value = "output"        
        response = self.app.post("/pbpk/httk", json={})
        
        response = self.app.post("/pbpk/httk", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, "output")
        
        
    def test_httk_invalid_params(self):
        response = self.app.post("/pbpk/httk", json={"chem_name": "Bisphenol A", "species": "human", "daily_dose": -1, "doses_per_day": 1, "days": 15})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "400 Bad Request: daily_dose must be a positive number"})
        
    @patch("app.run_r_model")
    def test_httk_run_r_model_exception(self, mock_run_r_model):
        mock_run_r_model.side_effect = Exception("Run R Model Error")
        response = self.app.post("/pbpk/httk", json={"chem_name": "Bisphenol A", "species": "human", "daily_dose": 1, "doses_per_day": 1, "days": 15})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Internal server error", response.get_data(as_text=True))
        
    # run_r_model tests
    @patch("rpy2.robjects.conversion.get_conversion")
    @patch("rpy2.robjects.globalenv")
    @patch("rpy2.robjects.r.source")
    def test_run_r_model_valid(self, mock_source, mock_globalenv, mock_get_conversion):
        mock_get_conversion.return_value = None
        mock_source.return_value = None
        mock_globalenv.__getitem__.return_value = MagicMock()
        
        output = run_r_model("script_path", "function_name", "arg1", "arg2")
        self.assertEqual(output, [])
        
    @patch("rpy2.robjects.conversion.get_conversion")
    @patch("rpy2.robjects.globalenv")
    @patch("rpy2.robjects.r.source")
    def test_run_r_model_exception(self, mock_source, mock_globalenv, mock_get_conversion):
        mock_get_conversion.return_value = None
        mock_source.side_effect = Exception("Run R Model Error")
        mock_globalenv.__getitem__.return_value = MagicMock()
        
        output = run_r_model("script_path", "function_name", "arg1", "arg2")
        self.assertEqual(output, None)
        
    @patch("rpy2.robjects.conversion.get_conversion")
    @patch("rpy2.robjects.globalenv")
    @patch("rpy2.robjects.r.source")
    def test_run_r_model_no_conversion(self, mock_source, mock_globalenv, mock_get_conversion):
        mock_get_conversion.return_value = None
        mock_source.return_value = None
        mock_globalenv.__getitem__.return_value = MagicMock()
        
        output = run_r_model("script_path", "function_name", "arg1", "arg2")
        self.assertEqual(output, [])
        
    @patch("rpy2.robjects.conversion.get_conversion")
    @patch("rpy2.robjects.globalenv")
    @patch("rpy2.robjects.r.source")
    def test_run_r_model_no_function(self, mock_source, mock_globalenv, mock_get_conversion):
        mock_get_conversion.return_value = None
        mock_source.return_value = None
        mock_globalenv.__getitem__.side_effect = KeyError("Function not found")
        
        output = run_r_model("script_path", "function_name", "arg1", "arg2")
        self.assertEqual(output, None)
        
    @patch("rpy2.robjects.conversion.get_conversion")
    @patch("rpy2.robjects.globalenv")
    @patch("rpy2.robjects.r.source")
    def test_run_r_model_no_source(self, mock_source, mock_globalenv, mock_get_conversion):
        mock_get_conversion.return_value = None
        mock_source.side_effect = FileNotFoundError("File not found")
        mock_globalenv.__getitem__.return_value = MagicMock()
        
        output = run_r_model("script_path", "function_name", "arg1", "arg2")
        self.assertEqual(output, None)

if __name__ == "__main__":
    unittest.main()
