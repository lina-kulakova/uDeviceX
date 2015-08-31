#!/usr/bin/awk -f

# Takes lines in the form
# `id', force, degree
# and produces a body of `check_pid_stretching'

{
  sub(/#.*/, "") # strip comments
}

NF {
  id = $1
  force = $2
  degree = $3
  # number of angles this atom is in
  ang    = $4

  id0 = id

  # first line is special
  if (NR==1) {
    printf "// this file was generated with configs/gen_check/gen_check.sh\n"
    printf "// `ang' is a number of angles atom is in\n"
    printf "if      (pid == %d) return %12.6gf; // ang: %d\n", id0, force, ang
  } else
    printf "else if (pid == %d) return %12.6gf; // ang: %d\n", id0, force, ang
}

END {
  # last line
  printf "else                 return %12.6ff;\n", 0.0
}
