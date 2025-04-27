# Base stage with system dependencies
FROM python:3.9-slim AS system-deps

# Install system packages
RUN apt-get update && \
    apt-get install -y wget coreutils libxrender1 libxext6 make gcc libpcre2-dev \
    liblzma-dev libbz2-dev zlib1g-dev build-essential libicu-dev libblas-dev libstdc++6 gnupg2 \
    libzstd-dev  # Add this package

# R dependencies stage
FROM system-deps AS r-deps

# Setup R repository
RUN echo "deb http://cloud.r-project.org/bin/linux/debian bookworm-cran40/" >> /etc/apt/sources.list && \
    gpg --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' && \
    gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | tee /etc/apt/trusted.gpg.d/cran_debian_key.asc

# Install R and packages
RUN apt-get update && apt-get install -y r-base
RUN R -e "install.packages(c('deSolve', 'msm', 'data.table', 'survey', 'mvtnorm', 'truncnorm', 'magrittr', 'purrr', 'Rdpack', 'mrgsolve', 'stringr'), repos='http://cran.us.r-project.org')"

# Python/Conda dependencies stage
FROM r-deps AS conda-deps

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda3 && \
    echo "export PATH=$HOME/miniconda3/bin:$PATH" >> ~/.bashrc

ENV PATH="/root/miniconda3/bin:${PATH}"

# Configure conda
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda install -c conda-forge libstdcxx-ng=14.1.0 -y

# CDDD environment setup
WORKDIR /app/models/PipelineAlternative_clinicaldata/cddd
COPY ./models/PipelineAlternative_clinicaldata/cddd/environment.yml ./
RUN conda env create -f environment.yml && \
    conda install -c conda-forge libstdcxx-ng -y
RUN ~/miniconda3/envs/cddd/bin/python -m pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

# Final stage - only copy application code here
FROM conda-deps AS final

WORKDIR /app

# Copy R package first (needed for installation)
COPY ./models/PBPK/httk/modified_package_tar/httkfb2.tar /app/models/PBPK/httk/modified_package_tar/
RUN R -e "install.packages('/app/models/PBPK/httk/modified_package_tar/httkfb2.tar', repos = NULL, type = 'source')"
RUN R -e 'install.packages("httk", repos="https://cloud.r-project.org/")'

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy CDDD code and install
COPY ./models/PipelineAlternative_clinicaldata/cddd /app/models/PipelineAlternative_clinicaldata/cddd
RUN cd /app/models/PipelineAlternative_clinicaldata/cddd && ~/miniconda3/envs/cddd/bin/python -m pip install -e .

# Copy the rest of your application code last
# This is the part that will change most frequently
COPY . .

EXPOSE 5000
CMD ["gunicorn", "-b", ":5000", "app:app"]