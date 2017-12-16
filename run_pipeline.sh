#!/bin/bash

#sgd_name=$1
#initial_learning_rate=0 # -1 Indicates to use custom optimal learning rate for Eurlex for each algorithm
proportion_data=0.1


for sgd_name in 'nce' 'sampled_softmax' #'Implicit' 'Umax' 'VanillaSGD'#
do
    for dataset_name in 'wikiSmall' # 'AmazonCat' # 'mnist' #'Eurlex' #'Bibtex' #'Delicious' 'wiki10'
    do

    # If want to iterate over multiple learning rates
        for initial_learning_rate in 3 2 1 0 -1 -2 -3
        do
            python3 pipeline.py $sgd_name $dataset_name $initial_learning_rate $proportion_data
        done

    done
done

python3 pipeline.py ove wikiSmall 3 1