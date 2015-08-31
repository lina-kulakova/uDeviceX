# parameters of the simulations

# this is linked to mpi-dpd/common.h
# and should be changed simultaneously
XSIZE_SUBDOMAIN=8
YSIZE_SUBDOMAIN=8
ZSIZE_SUBDOMAIN=8

xranks=1
yranks=1
zranks=1
stretching_force=100.0
tend=50

args="$xranks $yranks $zranks -rbcs -stretching_force=$stretching_force -tend=$tend"
