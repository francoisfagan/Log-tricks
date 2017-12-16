#!/bin/bash

python3 pipeline.py Implicit mnist -1 1
python3 pipeline.py Implicit Bibtex 1 1
python3 pipeline.py Implicit Delicious -2 1
python3 pipeline.py Implicit AmazonCat -3 1
python3 pipeline.py Implicit wiki10 0 1
python3 pipeline.py Implicit wikiSmall -3 1