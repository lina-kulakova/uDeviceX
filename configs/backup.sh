#!/bin/bash

# backup all files given as arguments

suffix=_bak_
for f ; do
    org="$f"
    bak="$f".$suffix
    test -r "$org" && cp "$org" "$bak"
done
