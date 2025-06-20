# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from SWE-agent/docker/swe.Dockerfile
FROM ubuntu:jammy

ARG TARGETARCH

# Install third party tools
RUN apt-get update && \
    apt-get install -y bash gcc git jq wget g++ make file sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ! We don't need git for MLGym as of now
# Initialize git
# RUN git config --global user.email "sweagent@pnlp.org"
# RUN git config --global user.name "sweagent"

RUN groupadd -r mlgym
RUN useradd -r -g mlgym -m -d /home/agent -s /bin/bash agent

# add agent user to sudoers list for apt commands
RUN echo "agent ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt" >> /etc/sudoers.d/agent \
    && chmod 440 /etc/sudoers.d/agent

# Environment variables
ENV ROOT='/home/agent/'
RUN prompt() { echo " > "; };
ENV PS1="> "

# ! We also don't need to track edits for patch
# Create file for tracking edits, test patch
# RUN touch /root/files_to_edit.txt
# RUN touch /root/test.patch

WORKDIR /home/agent

# Install miniconda
ENV PATH="/home/agent/miniconda3/bin:${PATH}"
ARG PATH="/home/agent/miniconda3/bin:${PATH}"
ENV PATH="/home/agent/.local/bin:${PATH}"
ARG PATH="/home/agent/.local/bin:${PATH}"

# Install generic conda environment requirements
COPY docker/requirements.txt /home/agent/requirements.txt
COPY data/rlMountainCarContinuous/requirements.txt /home/agent/generic_conda_requirements.txt
COPY docker/setup_conda.sh /home/agent/setup_conda.sh
COPY docker/custom_bashrc /home/agent/.bashrc
RUN chmod +x /home/agent/setup_conda.sh

USER agent
RUN echo $USER
COPY docker/getconda.sh .
RUN bash getconda.sh ${TARGETARCH} \
    && rm getconda.sh \
    && mkdir /home/agent/.conda \
    && bash miniconda.sh -b -u -p /home/agent/miniconda3 \
    && rm -f miniconda.sh
RUN conda --version \
    && conda init bash \
    && . /home/agent/.bashrc \
    && conda config --append channels conda-forge

# Cache python versions
RUN conda create -y -n rlMountainCarContinuous python=3.11

# Install python packages
RUN pip install -r /home/agent/requirements.txt
RUN /home/agent/setup_conda.sh
# RUN /bin/bash -c 'conda init bash \
#     && source /home/agent/.bashrc \
#     && conda activate mlgym_generic \
#     && pip install -r /home/agent/generic_conda_requirements.txt'

CMD ["/bin/bash"]
