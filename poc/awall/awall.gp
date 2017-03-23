set param
set size sq

r = 1
f = "<awk '$10 == 1' d"
plot [0:2*pi] sin(t), cos(t), f, f u 4:5, f u 7:8
