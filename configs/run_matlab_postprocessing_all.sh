#!/bin/bash

filedir=$1

for fullpath in `find ~ -maxdepth 1 -type d -name '*gammadpd_5*'`; do
	dirname=`basename ${fullpath}`

	echo "Processing ${fullpath}"

	cp dpd_kolmogorov.m ${fullpath}

	cd ${fullpath}
	matlab -singleCompThread -nodisplay -r "try dpd_kolmogorov('${filedir}','${dirname}'); end; exit()"
	cat "results.txt"
	cd -
done
