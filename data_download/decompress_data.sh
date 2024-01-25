#!/bin/bash

# Decompress and move
RAW_DATA="./tar_data"
DATA="./marine_data"

find $RAW_DATA -name "*.tar.1.gz" -exec tar -xzvf {} -C $DATA \;
