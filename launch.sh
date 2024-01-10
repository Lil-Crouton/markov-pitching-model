#!/bin/bash

NAME=markov_app

jupyter nbconvert --to script $NAME.ipynb

if [ -f $NAME.txt ]; then
    mv $NAME.txt $NAME.py
fi