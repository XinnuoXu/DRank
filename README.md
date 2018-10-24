# msr_dialog_ranking

After releasing their dialogue system (DS) to users, developers can improve the system’s performance by correcting mistakes the system made in dialogues and retraining the DS model on the labeled data. However, going through all dialogues is time consuming. So we need a ranker to detect dialogues with lower quality automatically to make this dialogue learning process with human-in-the-loop efficient. Also, gathering labeled data for supervised learning is time consuming. Therefore, we are building a unsupervised pointwise neural ranker.

## References <br />

This code is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)

## Quickstart <br />

### Environment set up
Since references are based on different environment, we use [Miniconda](https://conda.io/miniconda.html) to set up environment for each regerence. Please run the following codes to set up miniconda.

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.bashrc
echo ". ~/miniconda2/etc/profile.d/conda.sh" >> ~/.benv
source ~/.benv
```

Run `conda --help` to test.

### Supervised learning <br />

#### AMT data collection <br />
AMT data collection is working in `AMT/`. 

```
conda create -n AMT python=3
```
We will be working with some example data in `data/` folder. The data consists of parallel dialogue context (`.en`) and its response (`.vi`) data containing one sentence per line with tokens separated by a space:

* `train.en`
* `train.vi`
* `dev.en`
* `dev.vi`

After running the preprocessing, the following files are generated in `data/` folder:

* `dialogue.train.1.pt`: serialized PyTorch file containing training data
* `dialogue.valid.1.pt`: serialized PyTorch file containing validation data
* `dialogue.vocab.pt`: serialized PyTorch file containing vocabulary data, which will be used in the training process of language model.

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
