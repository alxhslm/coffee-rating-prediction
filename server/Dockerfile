FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app

# Copy the Pip files into our working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker
COPY ["predict.py", "model.bin", "./"]

# We need to expose the 8501 port because we're not able to communicate with Docker outside it
EXPOSE 8501

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8501", "predict:app"]
