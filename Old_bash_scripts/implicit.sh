#!/bin/bash

python3 pipeline.py Implicit mnist -1 1 97 &
python3 pipeline.py Implicit Bibtex 1 1 68 &
python3 pipeline.py Implicit Delicious -2 1 57 &
python3 pipeline.py Implicit Eurlex 1 1 106 &
python3 pipeline.py Implicit AmazonCat -3 1 88 &
python3 pipeline.py Implicit wiki10 0 1 78 &
python3 pipeline.py Implicit wikiSmall -3 1 66 &