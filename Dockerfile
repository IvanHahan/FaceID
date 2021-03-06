FROM python:3.7.5-stretch
MAINTAINER 'ivanh@aiknights.com'

WORKDIR face_id

RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0/cmake-3.19.0-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"
RUN git clone https://github.com/davisking/dlib.git; cd dlib; mkdir build; cd build; cmake ..; cmake --build .

COPY requirements.txt .
RUN pip install --upgrade pip; \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000/tcp

ENV CONFIG='Debug'
ENV MODEL_PATH='/models/b0f8d69be9b04052ab1ad91d127f724a/artifacts/model/artifacts/swa.ckpt'
ENV KNOWN_FACES_DIR='/data/16.11.20'

ENTRYPOINT [ "gunicorn", "app:app", "--timeout=120", "-b=0.0.0.0:5000" ]