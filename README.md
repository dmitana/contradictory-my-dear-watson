# Contradictory, My Dear Watson
Contradictory, My Dear Watson is a [Kaggle competition](https://www.kaggle.com/c/contradictory-my-dear-watson/overview)
focused at a Natural Language Inference (NLI).
The goal is to predict whether one sentence entails, contradicts or si unrelated with the other.

## Prerequisites 
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

Native GPU support has not landed in docker-compose yet. For now install patched versions of `docker-py` and `docker-compose` as mentioned [here](https://github.com/docker/compose/issues/6691#issuecomment-571309691):
```bash
pip install --user git+https://github.com/docker/docker-py.git
pip install --user git+https://github.com/yoanisgil/compose.git@device-requests
```

## Getting Started
1. Build docker image:
```bash
$ COMPOSE_API_VERSION=auto docker-compose up --build -d
```

2. Docker container is running after building. Next time, docker container can be started and stopped as follows:
```bash
# Start docker container
$ docker-compose start

# Stop docker container
$ docker-compose stop
```

3. Download data:
```bash
$ kaggle competitions download -c contradictory-my-dear-watson
```

## Training
To train a new model create [run configuration](runs/) a start training:
```bash
$ python -m contradictory_my_dear_watson train @runs/<run.conf>
```

## Evaluation
To evaluate a trained model on test dataset execute:
```bash
$ python -m contradictory_my_dear_watson evaluate @run/<run.conf> --checkpoint-path models/<model.pt> --test-data-path data/<test.csv>
```

## Results
Due to time constraints I used only baseline `BiLSTM` [1] and `BERT multilingual` [2] model.
`BiLSTM` model was used only english data.
I did not do any text preprocessing, pretraining, data augmentation or hyperparameter optimization. 

Achieved accuracy using `BiLSTM` model is 51.53% and using `BERT multilingual` model is 67%. 
I know, not great, not terrible, but time ...


## References
[1] Conneau, A., et al. Supervised Learning of Universal Sentence Representations
from Natural Language Inference Data. arXiv preprint arXiv:1705.02364 (2018).

[2] Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding. In Proceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and Short Papers) (2019), Association
for Computational Linguistics, pp. 4171–4186.
