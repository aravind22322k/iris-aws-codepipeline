FROM apache/airflow:2.10.3

COPY requirements.txt /requirements.txt

RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install --no-cache-dir -r /requirements.txt

