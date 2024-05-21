# AI/ML API

This is a API server which provides endpoints for AI/ML models.

## Requirements

- Python 3.9
- PipelineAlternative_clinicaldata data
- cddd data

## Installation

Put the `PipelineAlternative_clinicaldata` and `cddd` directories in the root of the project.

Copy the content of `patches` directory into the `PipelineAlternative_clinicaldata` directory.

```sh
cp -r patches/* PipelineAlternative_clinicaldata/
```

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

## Testing

To run the tests:

```sh
python -m unittest app_test.py
```