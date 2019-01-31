FROM tensorflow/tensorflow:latest-py3
MAINTAINER Mayank Sanganeria (mayank@e7mac.com)
RUN apt-get update
RUN apt-get install --yes git vim
RUN git clone https://github.com/e7mac/ml-art
RUN pip install jupyter
EXPOSE 8888