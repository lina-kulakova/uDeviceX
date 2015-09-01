#!/bin/bash

filter_file=$1
parameter_file=$2

./filter.generator1.awk "$filter_file"
./filter.generator2a.awk "$filter_file" | ./filter.generator2b.awk

./filter.generator5a.awk   "$parameter_file"

./filter.generator2a.awk "$filter_file" | ./filter.generator3.awk
./filter.generator2a.awk "$filter_file" | ./filter.generator4.awk

./filter.generator5b.awk   "$parameter_file"


