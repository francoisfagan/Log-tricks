#!/bin/bash

python3 pipeline.py ove wikiSmall 3 1 &
python3 pipeline.py nce wikiSmall ? 1 &
python3 pipeline.py sampled_softmax wikiSmall ? 1 &