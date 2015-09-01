#!/bin/bash

filter_file=$1
parameter_file=$2


slave=`mktemp /tmp/slave.XXXXX`

./filter.generator.sh $filter_file $parameter_file > "$slave"

awk -f "$slave" "$parameter_file"

#echo "$slave"
