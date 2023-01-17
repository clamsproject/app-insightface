FROM python:3.9.12

WORKDIR ./app

COPY ./ ./

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

CMD ["python", "app.py"]
