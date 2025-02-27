FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update
RUN apt-get -y install tmux

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt