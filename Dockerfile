FROM python:3.7
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 20000-63000
CMD [ "python", "./server.py", "--ignore_off_track" ]