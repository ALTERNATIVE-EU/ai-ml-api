# Use an official Python runtime as a parent image
FROM python:3.9-slim AS base

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN rm -rf /app/.git

# Install system dependencies with retry logic
RUN apt-get update && \
    for i in {1..3}; do \
        apt-get install -y wget coreutils libxrender1 libxext6 make gcc libpcre2-dev \
        liblzma-dev libbz2-dev zlib1g-dev build-essential libicu-dev libblas-dev libstdc++6 gnupg2 && break || sleep 15; \
    done

# Set up R repository
RUN echo "deb http://cloud.r-project.org/bin/linux/debian bookworm-cran40/" >> /etc/apt/sources.list && \
    for i in {1..3}; do \
        gpg --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' && break || sleep 15; \
    done && \
    gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | tee /etc/apt/trusted.gpg.d/cran_debian_key.asc

# Install R and R packages with retry logic
RUN apt-get update && \
    for i in {1..3}; do \
        apt-get install -y r-base && break || sleep 15; \
    done

RUN for i in {1..3}; do \
        R -e "install.packages(c('deSolve', 'msm', 'data.table', 'survey', 'mvtnorm', 'truncnorm', 'magrittr', 'purrr', 'Rdpack', 'mrgsolve', 'stringr'), repos='http://cran.us.r-project.org')" && break || sleep 15; \
    done

RUN R -e "install.packages('/app/models/PBPK/httk/modified_package_tar/httkfb2.tar', repos = NULL, type = 'source')"

# Install Miniconda with retry logic
RUN for i in {1..3}; do \
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh && break || sleep 15; \
    done && \
    bash ~/miniconda.sh -b -p $HOME/miniconda3 && \
    echo "export PATH=$HOME/miniconda3/bin:$PATH" >> ~/.bashrc

ENV PATH="/root/miniconda3/bin:${PATH}"

# Configure conda channels
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Install conda packages with retry logic
RUN for i in {1..3}; do \
        conda install -c conda-forge libstdcxx-ng=14.1.0 -y && break || sleep 15; \
    done

# Set up CDDD environment
WORKDIR /app/models/PipelineAlternative_clinicaldata/cddd
RUN conda env create -f environment.yml && \
    conda install -c conda-forge libstdcxx-ng -y

RUN for i in {1..3}; do \
        ~/miniconda3/envs/cddd/bin/python -m pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl && break || sleep 15; \
    done && \
    ~/miniconda3/envs/cddd/bin/python -m pip install -e .

WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":5000", "app:app"]