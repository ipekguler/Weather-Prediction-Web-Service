#FROM mageai/mageai:latest
FROM python:3.10-slim

# Note: this overwrites the requirements.txt file in your new project on first run. 
# You can delete this line for the second run :) 
COPY requirements.txt ./requirements.txt

RUN pip3 install --force-reinstall pip==20.0.2
RUN pip install -r ./requirements.txt