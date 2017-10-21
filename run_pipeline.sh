#!/bin/bash

sgd_name=$1
#dataset_name='Bibtex'
initial_learning_rate=0 # -1 Indicates to use custom optimal learning rate for Eurlex for each algorithm
for dataset_name in 'Bibtex' 'Delicious' 'wiki10' 'mnist' # 'Eurlex' #'AmazonCat'  'wikiSmall' 'Bibtex' 'Delicious' 'wiki10' 'mnist'
do

## If want to iterate over multiple learning rates
#    for initial_learning_rate in 6 5 4 #3 2 1 0 -1 -2 -3 -4 -5 -6
#    do
#        python3 pipeline.py $sgd_name $dataset_name $initial_learning_rate $custom_learning_rate
#    done

    python3 pipeline.py $sgd_name $dataset_name $initial_learning_rate

done