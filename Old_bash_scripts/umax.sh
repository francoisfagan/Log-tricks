#!/bin/bash

python3 pipeline.py Umax mnist 1 1 91 &
python3 pipeline.py Umax Bibtex -1 1 74 &
python3 pipeline.py Umax Delicious -2 1 61 &
python3 pipeline.py Umax Eurlex -1 1 100 &
python3 pipeline.py Umax AmazonCat -2 1 91 &
python3 pipeline.py Umax wiki10 -2 1 86 &
python3 pipeline.py Umax wikiSmall -3 1 72 &