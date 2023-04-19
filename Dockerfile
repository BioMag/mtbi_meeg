# Dockerfile

FROM python:3.8-slim

# update & upgrade apt packages
RUN apt-get update -y && apt-get upgrade -y

# don't write .pyc files into image to reduce image size 
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# run the container as a non-root user
ENV USER=mtbi_meeg
RUN groupadd -r $USER && useradd -r -g $USER $USER

# install Python packages to venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt /
RUN python3 -m pip install -U pip
RUN python3 -m pip install -r requirements.txt

# copy application contents and install from source
COPY . .
RUN python3 -m pip install .

# switch to a non-root user
USER $USER

# change work directory
WORKDIR / 

# run bash
CMD ["/bin/bash"]