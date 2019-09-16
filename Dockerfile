FROM ubuntu:bionic

# Update base container install
RUN apt-get update
RUN apt-get upgrade -y

# Install GDAL dependencies
RUN apt-get install -y python3-pip locales

# Install dependencies for other packages
RUN apt-get install gcc g++
#RUN apt-get install jpeg-dev zlib-dev

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

RUN apt-get -y install zip
RUN apt-get install ca-certificates 
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Copy function to a path
RUN mkdir -p /var/car_prediction
COPY . /var/car_prediction/

# Work Directory
WORKDIR /var/car_prediction/

# Build context
ADD app.py src /

ENV PYTHONUNBUFFERED = '1'

# Upgrading pip
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel

# Install dependencies for tiling
RUN pip install -r requirements.txt

EXPOSE 4000

# CMD ["python3", "app.py" ]    
# CMD ["gunicorn", "-k", "gevent", "-w", "8", "-b", "0.0.0.0:4000", "wsgi:app"]
CMD ["gunicorn", "-c", "gconfig.py", "wsgi:app"]