#!/bin/bash

python3 pipeline.py nce mnist 1 1
python3 pipeline.py nce Bibtex 2 1
python3 pipeline.py nce Delicious ? 1
python3 pipeline.py nce AmazonCat ? 1
python3 pipeline.py nce wiki10 ? 1
python3 pipeline.py nce wikiSmall ? 1