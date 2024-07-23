FROM python:3.9.7-slim

COPY requirements.txt ./requirements.txt

RUN pip3 install --force-reinstall pip==20.0.2
RUN pip install -r ./requirements.txt

COPY [ "predict.py", "test.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]