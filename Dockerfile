FROM python:3.11-slim
WORKDIR /root

RUN apt-get update
RUN apt-get -y install curl
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN python -m pip install 'tensorflow==2.11.0''pandas' 'numpy'

RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code
RUN mkdir /root/trainer
COPY trainer/train.py /root/trainer/train.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3.11", "Chicago_trainer.py"]
