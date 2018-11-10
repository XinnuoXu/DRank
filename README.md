# msr_dialog_ranking

After releasing their dialogue system (DS) to users, developers can improve the system’s performance by correcting mistakes the system made in dialogues and retraining the DS model on the labeled data. However, going through all dialogues is time consuming. So we need a ranker to detect dialogues with lower quality automatically to make this dialogue learning process with human-in-the-loop efficient. Also, gathering labeled data for supervised learning is time consuming. Therefore, we are building a unsupervised pointwise neural ranker.

## References <br />

This code is based on

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/README.md)

## Quickstart <br />

### Environment set up
Since references are based on different environment, we use [Miniconda](https://conda.io/miniconda.html) to set up environment for each regerence. Please run the following codes to set up miniconda. After the installation, you can run `conda --help` to test.

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.bashrc
echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.benv
source ~/.benv
```

### AMT data collection <br />
We are using Amazon Mechanical Turk(AMT) to collect data for our experiments. We ask users to talk with Pydial restaurant finding bot to achieve a given goal, label the goal is achieved or not and the contextually wrong turns. 

AMT data collection is working in folder `AMT/`. You need to create a conda enviornment first by running

```
conda create -n AMT python=3
conda activate AMT
```

AMT data collection is based on [ParlAI](http://parl.ai/static/docs/mturk.html). To install ParlAI, please run the following command in folder `AMT/`.

```
python setup.py develop
```

To start the data collection, you need to run `sh run.sh` in folder `AMT/parlai/mturk/tasks/pydail_collection/`. If you run the script for the first time, it will ask for `Access Key ID` and `Secret Access Key`. Then, register your email address at [Heroku](https://signup.heroku.com/) and run `/home/xxu/msr_dialog_ranking/AMT/parlai/mturk/core/heroku-cli-v6.99.0-ec9edad-linux-x64/bin/heroku login` at the terminal to login to Heroku. The `Access Key ID` and `Secret Access Key` are using are

```
Access Key ID: AKIAIE57EGXOXHCR3TWA
Secret Access Key: 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
```

If you see the following information when you run `sh run.sh`, it means that you setup everything correctly. You can visit the `Link to HIT` to view your task. 

```
Creating HITs...
Link to HIT: https://workersandbox.mturk.com/mturk/preview?groupId=36Y7KK4GODJC2A2MECPHDQWE7L9SUZ

Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)

Local: Setting up WebSocket...
WebSocket set up!
```

Run the following scripts to process the raw data collected from AMT.  

```
python log_reading.py > AMT_dials.txt
python log_preprocess.py
```

After running this preprocessing, the following files are generated. `AMT_preprocessed.txt` is a file with all dialogues and rated scores. `generated_dial_examples_*` are randomly sampled positive and negative examples for supervised learning that we will explain later.

* `AMT_preprocessed.txt`
* `generated_dial_examples_dev.neg`
* `generated_dial_examples_dev.pos`
* `generated_dial_examples_test.neg`
* `generated_dial_examples_test.pos`
* `generated_dial_examples_train.neg`
* `generated_dial_examples_train.pos`


### Supervised model <br />
Based on the data collected from AMT, we train a supervised learning model to predict the quality of dialogues. The model is multi-heads self-attention with position embedding.

![image](https://github.com/XinnuoXu/msr_dialog_ranking/blob/master/supervised_learning.png)

Supervised learning is working in folder `supervised_learning/discriminator` based on [AllenNLP](https://allennlp.org/tutorials). You need to create a conda enviornment first by running

```
conda create -n ALLENNLP python=3.6
conda activate ALLENNLP
pip install allennlp=0.5.0
```

Training, validation, testing data are in folder `coherence/dataset_readers`. 

* `generated_dial_examples_dev.neg`
* `generated_dial_examples_dev.pos`
* `generated_dial_examples_test.neg`
* `generated_dial_examples_test.pos`
* `generated_dial_examples_train.neg`
* `generated_dial_examples_train.pos`

To train a model, run

```
sh discriminator_pretrain.sh
```

The trained model will be saved as `trained_models/hierarchical_coherence_attention/model.tar.gz`. To test it, run

```
discriminator_test.sh
```

You can change the hyper-parameters in `experiments/dialogue_context_hierarchical_coherence_attention_classifier.json`

### Unsupervised model <br />

The goal of unsupervised model is that given only gold dialogues (references during the DS construction) and the DS and giving a ranker that can detect user-to-system dialogues with lower quality. Our approach is (1) train a user simulator to talk with Pydail restaurant finding bot and collect conversations as negative examples (2) take gold dialogues as positive examples (3) train a discriminator to detect dialogues with lower quality.

#### Pre-training of seq2seq user simulator <br />

We pre-train our model on multi-domain Maluuba data. You can find Maluuba data and the script for format transfering in folder `Maluuba_data`. To read the original Maluuba data and transfer it into data for seq2seq model, you can run

```
mkdir data; python readline.py
```

After running this script, the following files are generated in folder `data/`

* `dev.en`
* `dev.vi`
* `test.en`
* `test.vi`
* `train.en`
* `train.vi`

#### Gold dialogue generation <br />

We generate 100 Gold dialogues by rule-based user simulator and restaurant finding system offered in Pydail. Gold dialogue generation is working in folder `gold_dialogue_generator/` based on [Pydial](http://www.camdial.org/pydial/). You need to create a conda enviornment first by running

```
TODO
```

To generate gold dialogues, run

```
sh generate.sh
```

you will find all generated dialogues in file `gold_dialogues.in`.

#### Model: Stepwised seq2seq user simulator (Stepwised-US) <br />

To generate negative examples for the training of discriminator, we use stepwised seq2seq as the user simulator and make it talk with the Pydail restaurant finding bot. During training process of user simulator, we save a model for each 6 epochs and collect 100 conversations between each model and Pydial restaurant finding bot. The positive examples for the training of discriminator is the duplication of gold dialogues.

Stepwised seq2seq user simulator is working in folder `stepwise_monster/` based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). You need to create a conda enviornment first by running

```
conda create -n GAN python=3.6
git clone https://github.com/OpenNMT/OpenNMT-py.git
pip install -r requirements.txt --user
```

We pre-train the seq2seq model on the Maluuba data mentioned before and train the model on gold dialogues. To pre-train the model, you need to copy data generated in folder `Maluuba_data` to folder `generator/data/` and run

```
sh pre-train.sh
```

You will find the pre-trained model in folder `dialogue-model`
