FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

# Install python packages
RUN pip install pip --upgrade
RUN pip install --no-cache-dir notebook \
	pandas matplotlib seaborn nltk numpy \
	torchtext spacy tqdm transformers tensorboard \
	torchsummaryX
RUN python -m spacy download en_core_web_sm

CMD jupyter notebook --ip 0.0.0.0 --port 8000 --allow-root
