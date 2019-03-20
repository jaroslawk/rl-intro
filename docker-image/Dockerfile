FROM jupyter/scipy-notebook:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python3 -m pip install --no-cache jupyterhub==0.9.4