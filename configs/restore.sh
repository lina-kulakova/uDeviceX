#!/bin/bash

# restore all files given as arguments

suffix=_bak_
for f ; do
    org="$f"
    bak="$f".$suffix
    test -r "$bak" && mv "$bak" "$org"
done
