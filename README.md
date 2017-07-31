BiDAF in Pytorch (work in progress)
---

## Requirements
- python 3.5.2
- pytorch 0.1.12
- numpy
- pandas
- msgpack
- spacy 1.x

## Quick Start
### Setup
- download the project via `git clone https://github.com/hitvoice/DrQA.git; cd DrQA`
- you may need to download spacy and compile from source. Cpython seems to fail with gcc 5.
- make sure python 3 and pip is installed.
- install [pytorch](http://pytorch.org/) matched with your OS, python and cuda versions.
- install the remaining requirements via `pip install -r requirements.txt`
- download the SQuAD datafile, GloVe word vectors and Spacy English language models using `bash download.sh`.

### Train

```bash
# prepare the data
python prepro.py
# train for 20 epoches with batchsize 32
python train.py -e 20 -bs 32
```
