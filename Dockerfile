FROM ubuntu:20.04

# https://northshorequantum.com/archives/dockerbuild_tz_hang.html
# Docker Build中に Configuring tzdataでハングするのを回避
ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    software-properties-common \
    locales \
    less \
    vim \
    git \
    wget \
    python3-pip \
    libgl1-mesa-dev

# settings for japanese
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TERM xterm

# install tesseract
RUN add-apt-repository ppa:alex-p/tesseract-ocr5
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev
# install tesseract fast model
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ \
    && wget -P /usr/share/tesseract-ocr/5/tessdata/ https://github.com/tesseract-ocr/tessdata_fast/blob/main/eng.traineddata

# able to use python command in addition
RUN ln -s /usr/bin/python3 /usr/bin/python
# install python library
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
