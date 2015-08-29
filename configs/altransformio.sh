#!/bin/bash

# TEST: altransformio1
# ./altransformio.sh test_data/transformio1.config a_42_c_10 > transformio.out.config
#
# TEST: altransformio2
# ./altransformio.sh test_data/transformio1.config c_10_a_42 > transformio.out.config
#
# TEST: altransformio3
# ./altransformio.sh test_data/transformio2.config a_2_c_2 > transformio.out.config
#
# TEST: altransformio4
# ./altransformio.sh test_data/transformio3.config a_1_b_2 > transformio.out.config
#
# TEST: altransformio5
# ./altransformio.sh test_data/transformio4.config a_1 > transformio.out.config

config_file=$1
parameters_line=$2

slave=`mktemp /tmp/altransf.XXXXXX`
awk -f altransformio.generator.awk "$config_file" "$parameters_line" > "$slave"
awk -f "$slave"
