FROM python:3.7
ADD server.py /
ADD src/ /
ADD media/ /
ADD commonroad-vehicle-models/ /
ADD requirements.txt /
RUN pip install -r requirements.txt
CMD [ "python", "./server.py" ]