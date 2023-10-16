FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["*.py", "model1.bin", "dv.bin", "./"]

RUN pipenv install --deploy --system


EXPOSE 9696

ENTRYPOINT [ "waitress-serve","--listen=0.0.0.0:9696","predict:app" ]