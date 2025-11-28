perl genfile.pl 1000000 input1.csv input2.csv

make

./hashtable.out 1000000 input1.csv input2.csv

Memory delay: 94.02

Table large: 0, Data large: 0

Present:

Hashed : 7.88 per cycle

Headers: 7.97 per cycle

Datas  : 7.91 per cycle

Cycles : 125640

Absent:

Hashed : 7.65 per cycle

Headers: 8.00 per cycle

Datas  : 0.37 per cycle

Cycles : 129558

present 12: 17.868 ns per key,  136.6 ns per cycle

absent  12: 17.224 ns per key,  131.7 ns per cycle


