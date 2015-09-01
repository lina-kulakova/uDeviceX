#!/bin/bash

dir=~/daint_synch/RBCstretching/numberdensity_10_dt_1e-3_kBT_0.404840_gammadpd_35_aij_0.25_stretchingforce_800_lmax_1.64_p_0.001412_cq_19.0476_kb_35_ka_2500_kv_3500_gammaC_50_totArea0_135_totVolume0_91

nd=`./ext.awk $dir | wc -l`

# use 10 as default
d=${1-10}

./ext.awk "$dir" | ./tail.awk -v nd=$nd -v d=$d | ./av.awk

../../allineario.awk `basename $dir`
