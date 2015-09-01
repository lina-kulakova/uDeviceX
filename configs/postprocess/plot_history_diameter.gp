set macros

basedirname="~/daint_synch/RBCstretching"
#basedirname="~/workspace/mounts/falcon/daint_synch/RBCstretching"

dir(aij,force) = sprintf("%s/numberdensity_10_dt_1e-3_kBT_0.404840_gammadpd_35_aij_%s_stretchingforce_%s_lmax_1.64_p_0.001412_cq_19.0476_kb_35_ka_2500_kv_3500_gammaC_50_totArea0_135_totVolume0_91", basedirname, aij, force);

file(aij,force) = sprintf("< grep \"RBC diameters:\" %s/rbc_stretching.*.o | awk \'{ print NR, $3, $4, $5}\'", dir(aij,force));

#plot [][] file("0.25","100") u 1:2 w lp, "" u 1:3 w lp, "" u 1:4 w lp, file("16","100") u 1:2 w lp, "" u 1:3 w lp, "" u 1:4 w lp

aij_str = "0.25 0.5 1 2 4 8 16"
w(s,i) = word(s,i)

f="800"
set key bottom
te = "set term x11 "
@te 1
plot [][7:12] for [i=1:words(aij_str)] file(w(aij_str,i), f) u 1:2 w lp t w(aij_str,i)
@te 2
plot [][4:9] for [i=1:words(aij_str)] file(w(aij_str,i), f) u 1:3 w lp t w(aij_str,i)
@te 3
plot [][2:7] for [i=1:words(aij_str)] file(w(aij_str,i), f) u 1:4 w lp t w(aij_str,i)
