# base image
FROM ubuntu:22.04

# installing fundamental packages
RUN apt-get update && apt-get install -y \
    vim \
    build-essential \
    cmake \
    gfortran \
    python3 \
    python3-pip \
    python3-venv \
    mpich \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python setting
RUN python3 -m pip install --upgrade pip

# setting a user
ARG USERNAME=kosuke
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# creating the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# switch to the created user
USER $USERNAME

# ADD a path for local pip installations
RUN echo "export PATH=$PATH:/home/$USERNAME/.local/bin" >> ~/.bashrc
