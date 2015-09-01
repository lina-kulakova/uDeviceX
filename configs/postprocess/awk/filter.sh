#!/bin/bash

# TEST: filter1
# ./filter.sh test_data/a.filter test_data/a[1-4].txt  > filter.out.txt
#
# TEST: filter2
# ./filter.sh test_data/ab.filter test_data/ab[1-4].txt  > filter.out.txt

filter_file=$1
shift

for f
do
    ./filter1.sh $filter_file $f
done
