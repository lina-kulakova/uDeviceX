set macro
set param
set size sq

r = 1.5
c = '[0:2*pi] r*sin(t), r*cos(t)'

f = "<awk '$10 == 2' d"

rv = 'f u (column(i)):(column(i+1))'

plot @c, i = 1, @rv, @rv
