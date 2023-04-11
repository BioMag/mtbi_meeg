FROM continuumio/miniconda3:latest

RUN conda update conda
RUN conda install -y -c conda-forge \
    python=3.8 \
    doit \
    scipy \
    mne>=1.3 \
    h5py>=3.8.0 \
    numpy>=1.20.3 \
    pandas>=1.5.2 \
    scikit-learn>=1.1.2 \
    matplotlib>=3.1.2 \
    weasyprint>=58.1

COPY . /app
WORKDIR /app

CMD ["python", "my_package.py"]

# to run it:
#    Build the Docker image: Run the command docker build -t my_package . to build the Docker image. This will create a new image named my_package based on the Dockerfile.

#    Run the Docker container: Run the command docker run -it my_package to start the container and run your package. This will start a new container based on the my_package image and run the default command specified in the Dockerfile.
