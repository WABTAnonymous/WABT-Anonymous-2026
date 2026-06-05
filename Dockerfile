FROM python:3.10-slim

WORKDIR usr/src/app

RUN pip install --no-cache-dir jupyter notebook

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the port Jupyter runs on
EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''