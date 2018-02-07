#!/bin/bash

python3 pipeline.py sampled_softmax mnist 1 1
python3 pipeline.py sampled_softmax Bibtex 2 1
python3 pipeline.py sampled_softmax Delicious 3 1
python3 pipeline.py sampled_softmax AmazonCat 3 1
python3 pipeline.py sampled_softmax wiki10 2 1
python3 pipeline.py sampled_softmax wikiSmall ? 1