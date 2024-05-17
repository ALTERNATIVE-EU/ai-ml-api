README: INFERENCE MODELS

## Requirements

- Python 3.9

How to use start the API server:

Install dependencies:

```sh
pip install -f requirements.txt
```

Start the server:

```sh
python3 app.py
```

Test the server:

ML:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/ml/evaluate -o results.csv
```

AI:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/ai/evaluate
```
