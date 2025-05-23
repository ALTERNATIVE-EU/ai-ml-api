# AI/ML API

This is a API server which provides endpoints for AI/ML models.

## Requirements

- Python 3.9
- Anaconda
- PipelineAlternative_clinicaldata data
- cddd data

## Installation

Copy the content of `patches` directory into the `models` directory.

```sh
cp -r patches/* ./models/
```

Create cddd virtual environment and install dependencies:

```sh
cd cddd
conda env create -f environment.yml
conda activate cddd
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
pip install -e .
conda deactivate
```

Create alternative virtual environment:

```sh
conda create -n alternative python=3.9
conda activate alternative
```

Install dependencies:

```sh
pip install -r requirements.txt
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

## Deployment in Kubernetes

Replace the `<image>` placeholder in `deployment/kubernetes/deployment.yaml` with your image name.

```sh
kubectl apply -f deployment/kubernetes
```

## Usage

### ML:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/clinicaldata/ml/evaluate -o results.csv
```

### AI:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/clinicaldata/ai/evaluate
```

### AOP:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/clinicaldata/aop/evaluate
```

### hERG
```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "c1ccccc1O"}' http://localhost:5000/clinicaldata/herg/evaluate
```

### Multitask
```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "C2C(N=Cc1ccccc1)=C(N(N2c3ccccc3)C)C"}' http://localhost:5000/clinicaldata/multitask/evaluate
```

### AHR
```sh
curl -X POST -H "Content-Type: application/json" -d '{"smiles": "C1=CC2=C(C(=C1)O)C(=O)C3=C(C2=O)C=C(C=C3O)CO"}' http://localhost:5000/clinicaldata/ahr/evaluate
```

Doxorubicin:

```sh
curl -X POST http://127.0.0.1:5000/pbpk/doxorubicin      -H "Content-Type: application/json"      -d '{
           "dose_mg": 60,
           "age": 50,
           "weight": 70,
           "height": 190
         }'
```

### HTTK:

```sh
curl -X POST http://127.0.0.1:5000/pbpk/httk -H "Content-Type: application/json"      -d '{
           "chem_name": "Bisphenol A",
           "species": "human",
           "daily_dose": 1,
           "doses_per_day": 1,
           "days": 15
         }'
```

TD:

### Proteomics:

```sh
curl -X 'POST' \
  'http://localhost:5000/TD/proteomics/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "compound": "DOX",
  "protein": "B8ZZL8"
}'

curl -X 'POST' \
  'http://localhost:5000/TD/proteomics/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "compound": "ROT",
  "protein": "D6RF35"
}'

curl -X 'POST' \
  'http://localhost:5000/TD/proteomics/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "compound": "AMI",
  "protein": "D6RF35"
}'
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