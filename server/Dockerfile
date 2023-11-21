FROM public.ecr.aws/lambda/python:3.11

RUN pip install pipenv

# Copy the Pip files into our working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker
COPY ["predict.py", "model.bin", "./"]

CMD [ "predict.lambda_handler" ]
RUN pip install pipenv
