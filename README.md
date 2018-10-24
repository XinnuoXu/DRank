# msr_dialog_ranking

After releasing their dialogue system (DS) to users, developers can improve the system’s performance by correcting mistakes the system made in dialogues and retraining the DS model on the labeled data. However, going through all dialogues is time consuming. So we need a ranker to detect dialogues with lower quality automatically to make this dialogue learning process with human-in-the-loop efficient. Also, gathering labeled data for supervised learning is time consuming. Therefore, we are building a unsupervised pointwise neural ranker.

## References <br />

This code is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)

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
AMT data collection is working in `AMT/`. We need to create a conda enviornment first by running

```
conda create -n AMT python=3; conda activate AMT
```

AMT data collection is based on [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/README.md). To install ParlAI, please run the following command in folder `AMT/`.

```
python setup.py develop
```

To start the data collection, you need to run `sh run.sh` in folder `AMT/parlai/mturk/tasks/pydail_collection/`. If you run the script for the first time, it will ask for `Access Key ID` and `Secret Access Key`. Then, register your email address at [Heroku](https://signup.heroku.com/) and run `/home/xxu/msr_dialog_ranking/AMT/parlai/mturk/core/heroku-cli-v6.99.0-ec9edad-linux-x64/bin/heroku login` at the terminal to login to Heroku. The `Access Key ID` and `Secret Access Key` we are using are

```
Access Key ID: AKIAIE57EGXOXHCR3TWA
Secret Access Key: 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
```

If you see the following information when you run `sh run.sh`, it means that you setup everything correctly. You can visit the `Link to HIT` to see your task. 

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
After running the preprocessing, the following files are generated. `AMT_preprocessed.txt` is a file with all dialogues and rated scores. `generated_dial_examples_*` are positive and negative examples for supervised learning that we will explain later.

* `AMT_preprocessed.txt`
* `generated_dial_examples_dev.neg`
* `generated_dial_examples_dev.pos`
* `generated_dial_examples_test.neg`
* `generated_dial_examples_test.pos`
* `generated_dial_examples_train.neg`
* `generated_dial_examples_train.pos`


### Step2: Train a language model <br />

```
cd lm/tool/
```

In this step, we will train a language model based on the responses for the MMI-anti model (example data `data/*.vi`). Since this language model will be used in the MMI-anti model, it will share the dictionary (`data/*.vocab.pt`) generated in `Step1`.

#### Step2.1: Preprocess the data <br /> 

```
python preprocess.py
```

These preprocessing will turn all responses for the MMI-anti model (example data `data/*.vi`) into parallel data for the language model. 


After running the preprocessing, the following files are generated in `lm/data/` folder:

* `train.en`
* `train.de`
* `dev.en`
* `dev.de`

For example, the response `"they just want a story"` in file `data/train.vi` will be preprocessed in to `"<s> they just want a story"` in file `lm/data/train.en` and `"they just want a story </s>"` in file `lm/data/train.de`.

#### Step2.2: Train a language model <br />

```
cd ../
python lm.py
```

This train command will save the language model to `lm/model.pt`.

To run this code on the CPU, you need to update your pytorch to any version after `24th Feb 2018` and make sure that this piece of code can be found in your `torchtext/data/iterator.py`:

```
if not torch.cuda.is_available() and self.device is None:
  self.device = -1
```

#### Step2.3: Test your language model <br />

```
python generate.py
```

This tool will generate 1000 utterances randomly using the language model `lm/model.pt` and save them into file `lm/generated.txt`.


#### Step2.4: Go back to our MMI-anti model <br />

```
cd ../
```

### Step3: Train a MMI-anti model <br />

```
python train.py
```

### Step4: Generate <br />

```
python translate.py -model model_name
```

The generation results will be saved in file `pred.txt`.

### Step5: Evaluate the diversity? <br />

```
cat pred.txt | python diversity.py
```
