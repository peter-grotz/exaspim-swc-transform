# hash:sha256:91f839cfd71c36413e37d912f36a591017edf610ce743e686d587825dd4120cd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    numpy==1.23.5 \
    allensdk==2.16.2 \
    git+https://github.com/AllenNeuralDynamics/aind_exaspim_register_cells.git
