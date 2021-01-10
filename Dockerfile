FROM python:3.7.5-stretch
MAINTAINER 'ivanh@aiknights.com'

WORKDIR face_id

RUN apt-get update; apt-get install -y software-properties-common; apt-get install -y snapd; snap install cmake -classic
RUN git clone https://github.com/davisking/dlib.git; cd dlib; mkdir build; cd build; cmake ..; cmake --build .

COPY requirements.txt .
RUN pip install --upgrade pip; \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000/tcp

ENV CONFIG='Debug'
ENV MODEL_PATH='models/b0f8d69be9b04052ab1ad91d127f724a/artifacts/model/artifacts/swa.ckpt'
ENV KNOWN_FACES_DIR='data/16.11.20'

ENTRYPOINT [ "gunicorn", "app:app", "--timeout=120", "-b=0.0.0.0:5000" ]