#!/bin/bash

python3 pipeline.py VanillaSGD mnist -2 1
python3 pipeline.py VanillaSGD Bibtex -2 1
python3 pipeline.py VanillaSGD Delicious -3 1
python3 pipeline.py VanillaSGD Eurlex -3 1
python3 pipeline.py VanillaSGD wiki10 -3 1