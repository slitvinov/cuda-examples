#! /bin/bash

set -e
set -u
for n in $(seq 1 10 | awk '{print $1*512}') ; do
    time ./sum_kernel ${n}
done
