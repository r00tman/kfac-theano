#!/usr/bin/python

import sys
import os

hdr = "gnuplot -p -e 'set term qt size 1600, 800; set logscale y 10; plot "
hdr = "gnuplot -p -e 'set term qt size 1600, 800; set xrange [*:1000]; set logscale y 10; plot "
# hdr = "gnuplot -p -e 'set term qt size 1600, 800; plot "
# hdr = "gnuplot -p -e 'set term qt size 1600, 800; set xrange [*:100]; set yrange [1e-11:*]; set logscale y 10; plot "
tpl = "\"{}\" using 0:{} with lines, "

res = hdr
c = sys.argv[1]
for p in c.split(','):
    for x in sys.argv[2:]:
        res += tpl.format(x, p)

res += "'"
os.system(res)
