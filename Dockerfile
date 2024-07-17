# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN rm -rf /app/.git

# Install Miniconda
RUN apt-get update && apt-get install -y wget coreutils libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda3
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda --version

WORKDIR cddd
RUN conda env create -f environment.yml
RUN ~/miniconda3/envs/cddd/bin/python -m pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
RUN ~/miniconda3/envs/cddd/bin/python -m pip install -e .

WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":5000", "app:app"]