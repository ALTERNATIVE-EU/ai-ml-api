import unittest
from flask import Flask, jsonify, request, send_file
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd
import json
import os

from app import app


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # /ml/evaluate tests
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.check_smiles")
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.import_models")
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.cddd_calculation")
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.ad_evaluation")
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.transform_data")
    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.inference")
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

        response = self.app.post("/ml/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Content-Disposition"], "attachment; filename=results.csv"
        )

    def test_ml_evaluate_no_smiles(self):
        response = self.app.post("/ml/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    def test_ml_evaluate_empty_smiles(self):
        response = self.app.post("/ml/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    @patch("PipelineAlternative_clinicaldata.ML_apical.inference.check_smiles")
    def test_ml_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/ml/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Check SMILES Error", response.get_data(as_text=True))

    # /ai/evaluate tests
    @patch("PipelineAlternative_clinicaldata.AI.inference.check_smiles")
    @patch("PipelineAlternative_clinicaldata.AI.inference.import_NLP_model")
    @patch("PipelineAlternative_clinicaldata.AI.inference.prepare_smiles_NLP")
    @patch("PipelineAlternative_clinicaldata.AI.inference.ad_evaluation")
    @patch("PipelineAlternative_clinicaldata.AI.inference.inference_NLP")
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

        response = self.app.post("/ai/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"prediction": "inference"})

    def test_ai_evaluate_no_smiles(self):
        response = self.app.post("/ai/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    def test_ai_evaluate_empty_smiles(self):
        response = self.app.post("/ai/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    @patch("PipelineAlternative_clinicaldata.AI.inference.check_smiles")
    def test_ai_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/ai/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Check SMILES Error", response.get_data(as_text=True))


    # /aop/evaluate tests
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.check_smiles")
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.import_models")
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.cddd_calculation")
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.ad_evaluation")
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.transform_data")
    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.inference")
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

        response = self.app.post("/aop/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Content-Disposition"], "attachment; filename=results.csv"
        )

    def test_aop_evaluate_no_smiles(self):
        response = self.app.post("/aop/evaluate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    def test_aop_evaluate_empty_smiles(self):
        response = self.app.post("/aop/evaluate", json={"smiles": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No SMILES input provided"})

    @patch("PipelineAlternative_clinicaldata.AOP_models.inference.check_smiles")
    def test_aop_evaluate_check_smiles_exception(self, mock_check_smiles):
        mock_check_smiles.side_effect = Exception("Check SMILES Error")
        response = self.app.post("/aop/evaluate", json={"smiles": "CCO"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Check SMILES Error", response.get_data(as_text=True))

    # /isalive tests
    def test_isalive(self):
        response = self.app.get("/isalive")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "alive"})


if __name__ == "__main__":
    unittest.main()
