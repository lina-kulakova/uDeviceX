#!/bin/bash

# backup all files given as arguments

suffix=_bak_
for f ; do
    org="$f"
    bak="$f".$suffix
    # do not overwrite backup
    test -r "$bak" || cp "$org" "$bak"
    test -r "$bak" && echo "(backup.sh) I see backup file: $bak"
done
