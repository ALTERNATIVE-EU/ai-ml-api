# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN rm -rf /app/.git

# Install Miniconda
RUN apt-get update && apt-get install -y wget coreutils libxrender1 libxext6 make gcc libpcre2-dev liblzma-dev libbz2-dev zlib1g-dev build-essential libicu-dev libblas-dev libstdc++6 gnupg2
RUN echo "deb http://cloud.r-project.org/bin/linux/debian bookworm-cran40/" >> /etc/apt/sources.list
RUN gpg --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'
RUN gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | tee /etc/apt/trusted.gpg.d/cran_debian_key.asc
RUN apt-get update && apt-get install -y r-base

# Install R packages
RUN R -e "install.packages(c('deSolve', 'msm', 'data.table', 'survey', 'mvtnorm', 'truncnorm', 'magrittr', 'purrr', 'Rdpack', 'mrgsolve', 'stringr'), repos='http://cran.us.r-project.org')"
RUN R -e "install.packages('/app/models/PBPK/httk/modified_package_tar/httkfb2.tar', repos = NULL, type = 'source')"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda3
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda --version
RUN conda install -c conda-forge libstdcxx-ng -y

WORKDIR models/PipelineAlternative_clinicaldata/cddd
RUN conda env create -f environment.yml
RUN conda install -c conda-forge libstdcxx-ng -y
RUN ~/miniconda3/envs/cddd/bin/python -m pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
RUN ~/miniconda3/envs/cddd/bin/python -m pip install -e .

WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":5000", "app:app"]
