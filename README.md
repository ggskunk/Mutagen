The idea of ​​the program is to mutate one private key from the previous puzzle to find the solution to the next one.

We take, for example, the key to the 19th puzzle 0000000000000000000000000000000000000000000000000000000000005749f and the address of the next one in hash160 : b907c3a2a3b27789dfb509b730dd47703c272868

and mutate its private key bits, this allows us to reduce the number of options for enumerating options

build : g++ -O3 -march=native -std=c++17 mutagen.cpp -lssl -lcrypto -lpthread -o mutagen

./mutagen -p [PUZZLE_NUMBER] -t [THREAD_COUNT]

Options

-p	Puzzle number to solve (20-68)	20
-t	Number of worker threads to use	CPU core count
-h	Show help message	

Examples
Solve puzzle 25 using all available CPU cores:

./mutagen -p 25 

Solve puzzle 25 using 8 threads:

./mutagen -p 25 -t 8

The program is in development mode. If anyone wants to help improve it, they are welcome.

As improvements are made, the repository will be updated


donation address : 
bc1qdwnxr7s08xwelpjy3cc52rrxg63xsmagv50fa8
