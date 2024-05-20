# AI/ML API

This is a API server which provides endpoints for AI/ML models.

## Requirements

- Python 3.9

## Installation

Install dependencies:

```sh
pip install -f requirements.txt
```

Start the server:

```sh
python3 app.py
```

## Installation with Docker

Build the image:

```sh
docker build -t ai-ml-api .
```

Run the container:

```sh
docker run -p 5000:5000 ai-ml-api
```

## Usage

ML:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/ml/evaluate -o results.csv
```

AI:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/ai/evaluate
```

AOP:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/aop/evaluate
```

IsAlive:

```sh
curl http://localhost:5000/isalive
```
