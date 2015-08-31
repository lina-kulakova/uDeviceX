#!/usr/bin/awk -f

# Find 
#
# Outputs:
# `line' `degree' `nangles' `

BEGIN {
  rbcfile = ARGV[1]
  ARGV[1] = "-"
}

{
  id = $1
  line = $0

  id_lst[++iatm] = id
  arr[iatm] = line
}

function count_bonds(id1, id2) {
  while (getline < rbcfile > 0 && $1 != "Bonds") ;
  getline        < rbcfile # skip empty line
  while (getline < rbcfile > 0 && NF) {
    # ids in the bond line
    id1=$3; id2=$4
    bnd[id1]++; bnd[id2]++
  }
  close(rbcfile)
}

function count_angles(id1, id2, id3) {
  while (getline < rbcfile > 0 && $1 != "Angles") ;
  getline        < rbcfile # skip empty line
  while (getline < rbcfile > 0 && NF) {
    # ids in the angles line
    id1=$3; id2=$4; id3=$5
    ang[id1]++; ang[id2]++; ang[id3]++
  }
  close(rbcfile)
}

function count_dihedrals(id1, id2, id3, id4) {
  while (getline < rbcfile > 0 && $1 != "Dihedrals") ;
  getline        < rbcfile # skip empty line
  while (getline < rbcfile > 0 && NF) {
    # ids in the dihedrals line
    id1=$3; id2=$4; id3=$5; id4=$6
    dih[id1]++; dih[id2]++; dih[id3]++; dih[id4]++
  }
  close(rbcfile)
}

# force numeric
function fn(x) {
    return x+0
}

END {

  count_bonds()
  count_angles()
  count_dihedrals()

  for (i=1; i<=iatm; i++) {
    id = id_lst[i]
    line = arr[i]
    print line, fn(bnd[id]), fn(ang[id]), fn(dih[id])
  }

}
