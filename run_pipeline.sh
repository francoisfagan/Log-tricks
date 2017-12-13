#!/bin/bash

sgd_name=$1
initial_learning_rate=0 # -1 Indicates to use custom optimal learning rate for Eurlex for each algorithm
proportion_data=0.1
for dataset_name in 'AmazonCat' 'wikiSmall' 'Delicious' 'wiki10' # 'mnist' #'Eurlex' #'Bibtex' #
do

# If want to iterate over multiple learning rates
    for initial_learning_rate in 3 2 1 0 -1 -2 -3
    do
        python3 pipeline.py $sgd_name $dataset_name $initial_learning_rate $proportion_data &
    done

done