FROM python:3.11-slim

RUN pip install poetry==1.6.0

RUN poetry config virtualenvs.create false

# Copy the poetry files into our working directory
COPY ["pyproject.toml", "poetry.lock", "./"]

# install the poetry dependencies for the project and deploy them.
RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY ["predict.py", "model.bin", "./"]

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

CMD [ "predict.lambda_handler" ]
