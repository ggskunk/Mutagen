#!/bin/bash

# Start ve End aralığı
START=73786976294838206464
END=295147905179352825855

while true; do
    # Python ile -k değerini oluştur
    K_VALUE=$(python3 -c "import random; print(random.randint($START, $END))")

    for F in {1..7}; do
        echo "Running: ./mutagen -p 68 -k $K_VALUE -f $F"
        ./mutagen -p 68 -k $K_VALUE -f $F
    done
done
