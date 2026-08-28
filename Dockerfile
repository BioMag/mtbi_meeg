FROM continuumio/miniconda3:latest

WORKDIR /app
COPY environment.yml requirements.txt setup.py ./
COPY src ./src
COPY tests ./tests

RUN conda env create --file environment.yml && conda clean --all --yes
SHELL ["conda", "run", "-n", "mtbi_meeg_conda", "/bin/bash", "-c"]
RUN python -m pip install --no-deps .

ENV MTBI_MPL_BACKEND=Agg
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mtbi_meeg_conda"]
CMD ["python", "-m", "pytest", "-q"]
