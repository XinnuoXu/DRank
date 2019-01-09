# msr_dialog_ranking

Although the data-driven approaches of some recent bot building platforms make it possible for a wide range of users to easily create dialogue systems, those platforms don't offer tools for quickly identifying which log dialogues contain problems. This is important since corrections to log dialogues provide a means to improve performance after deployment. A log dialogue ranker, which ranks problematic dialogues higher, is an essential tool due to the sheer volume of log dialogues that could be generated. However, training a ranker typically requires labelling a substantial amount of data, which is not feasible for most users. In this paper, we present a novel unsupervised approach for dialogue ranking using GANs and release a corpus of labelled dialogues for evaluation and comparison with supervised methods. The evaluation result shows that our method compares favorably to supervised methods without any labelled data. To the best of our knowledge, we are the first to introduce the dialogue ranking task with accompanying data and present an unsupervised approach for training a dialogue ranker.

## References <br />

This code is based on

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/README.md)

## Environment set up
Since references are based on different environment, we use [Miniconda](https://conda.io/miniconda.html) to set up environment for each regerence. Please run the following codes to set up miniconda. After the installation, you can run `conda --help` to test.

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.bashrc
echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.benv
source ~/.benv
```

## AMT data collection <br />
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


## Supervised model <br />
A dialogue ranker aims to assign higher scores to problematic dialogues than normal ones so that developers may quickly identify problematic dialogues in the ranked list of log dialogues.

<p align="center">
<img src="https://github.com/XinnuoXu/msr_dialog_ranking/blob/master/supervised_learning.png" height="400" width="350">
</p>

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

## Unsupervised model <br />

Training a ranker typically requires labelling a substantial amount of data and one might have to repeat this process whenever a significant change is made to the system's behavior. This is not feasible for most developers and motivates us to explore a set of unsupervised approaches. The core idea is that we learn a generative user simulator and have it talk with the bot to produce problematic dialogues. We then train a ranker with seed dialogues used as normal examples. 

### Maluuba dataset <br />

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

### Gold dialogue generation <br />

We generate 100 Gold dialogues by rule-based user simulator and restaurant finding system offered in Pydail. Gold dialogue generation is working in folder `gold_dialogue_generator/` based on [Pydial](http://www.camdial.org/pydial/). You need to create a conda enviornment first by running

```
TODO
```

To generate gold dialogues, run

```
sh generate.sh
```

you will find all generated dialogues in file `gold_dialogues.in`.

### Model: StepGAN <br />

To generate negative examples for the training of discriminator, we use stepwised seq2seq as the user simulator and make it talk with the Pydail restaurant finding bot. During training process of user simulator, we save a model for each 6 epochs and collect 100 conversations between each model and Pydial restaurant finding bot. The positive examples for the training of discriminator is the duplication of gold dialogues.

Stepwised seq2seq user simulator is working in folder `GAN_monster/` based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). You need to create a conda enviornment first by running

```
conda create -n GAN python=3.6
git clone https://github.com/OpenNMT/OpenNMT-py.git
pip install -r requirements.txt
conda install pytorch=0.4.1 cuda80 -c pytorch
pip install torchtext
pip install six
pip install allennlp=0.5.0
cd pydial
pip install -r requirements.txt
pip install numpydoc
```

<p align="center">
<img src="https://github.com/XinnuoXu/msr_dialog_ranking/blob/master/plot_gans.png" height="300" width="350">
</p>

We pre-train the seq2seq model on the Maluuba data mentioned before and train the model on gold dialogues. To pre-train the model, you need to copy data generated in folder `Maluuba_data` to folder `generator/data/` and run

```
sh pre-train.sh
```

You will find the pre-trained model in folder `dialogue-model`
