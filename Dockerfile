FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

# Install python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir notebook pandas matplotlib seaborn nltk numpy

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8000", "--allow-root"]
