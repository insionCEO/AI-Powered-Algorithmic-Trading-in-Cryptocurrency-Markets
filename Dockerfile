
FROM python:3.8-buster

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements_service.txt ./
RUN pip install --no-cache-dir -r requirements_service.txt
#RUN pip install pandas

# Bundle app source
COPY /prediction_service .

EXPOSE 80
#CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]
CMD ["python3", "app.py"]
