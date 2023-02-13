FROM python:3.8.0-alpine

RUN apt update && apt install -y \
    locales \
    less \
    vim \
    git \
    wget

# settings for japanese
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# install tesseract
RUN add-apt-repository ppa:alex-p/tesseract-ocr5
RUN apt update && apt install -y \
    tesseract-ocr \
    libtesseract-dev
# install tesseract fast model
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ \
    && wget -P /usr/share/tesseract-ocr/5/tessdata/ https://github.com/tesseract-ocr/tessdata_fast/blob/main/eng.traineddata

# install python library
COPY requirements.txt .
RUN pip install -r requirements.txt
