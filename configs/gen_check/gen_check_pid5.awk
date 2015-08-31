#!/usr/bin/awk -f
#
# Scale force according to degree
# sum(force*degree) should be zero

{
  sub(/#.*/, "") # strip comments
  force_idx = 2
}

NF {
  iatm++

  id = $1
  force = $(force_idx)
  degree = $3
  ang    = $4

  # we add "force" `ang' times in the code
  # here it should be scaled accordingly
  $(force_idx) = force/ang

  print
}

