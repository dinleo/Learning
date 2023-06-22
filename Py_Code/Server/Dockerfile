FROM python:3

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY drf_test .

EXPOSE 8001

ENTRYPOINT ["python", "manage.py", "runserver", "0.0.0.0:8001"]