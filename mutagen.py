import time
import multiprocessing
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import secp256k1 as ice

NUM_WORKERS = 26
TARGET_ADDRESS = "95a156cd21b4a69de969eb6716864f4c8b82a82a"

stop_event = multiprocessing.Event()
key_found = multiprocessing.Value('b', False)

# Original key (from puzzle 19)
KEY_67_HEX = "0000000000000000000000000000000000000000000000000000004b5f8303e9"

# Fix the upper 236 bits (zero bits)
KEY_67_BIN = bin(int(KEY_67_HEX, 16))[2:].zfill(256)
FIXED_BITS = KEY_67_BIN[:216]  # Fixed 236 bits
CHANGING_BITS = list(KEY_67_BIN[216:])  # Changeable bits (20 bits)


def mutate_fixed_bits(start_index, num_workers):
    for i, bit_indices in enumerate(combinations(range(40), 20)):
        if (i % num_workers) == start_index:
            mutated_bits = CHANGING_BITS[:]
            for index in bit_indices:
                mutated_bits[index] = '1' if mutated_bits[index] == '0' else '0'
           
            mutated_key_bin = FIXED_BITS + "".join(mutated_bits)
            mutated_key_hex = hex(int(mutated_key_bin, 2))[2:].zfill(64)
            yield mutated_key_hex


def check_key(worker_id, num_workers, keys_checked, lock):
    local_counter = 0 
    start_time = time.time()

    for mutated_key_hex in mutate_fixed_bits(worker_id, num_workers):
        if key_found.value or stop_event.is_set():
            return None

        generated_address = ice.privatekey_to_h160(0, True, int(mutated_key_hex, 16)).hex()

        # Check for partial match
        if not generated_address.startswith(TARGET_ADDRESS[:6]):
            continue

        if generated_address == TARGET_ADDRESS:
            with lock:
                if not key_found.value:
                    print(f"\n Key found {worker_id}: {mutated_key_hex} to address {TARGET_ADDRESS}!")
                    with open("key.txt", "w") as f:
                        f.write(f"Key found: {mutated_key_hex}\n")
                    key_found.value = True
                    stop_event.set()
            return mutated_key_hex
       
        local_counter += 1 
        if time.time() - start_time >= 300:
            with lock:
                keys_checked.value += local_counter
            local_counter = 0 
            start_time = time.time() 

    with lock:
        keys_checked.value += local_counter 

    return None 


def print_status(start_time, keys_checked, lock):
    while not stop_event.is_set():
        time.sleep(300)
        with lock:
            checked = keys_checked.value 
        elapsed_time = time.time() - start_time
        print(f"Full time: {elapsed_time:.2f} sec. | Total Key: {checked}")


def main():
    start_time = time.time()

    with multiprocessing.Manager() as manager:
        keys_checked = manager.Value('i', 0)
        lock = manager.Lock()

        tracker_process = multiprocessing.Process(target=print_status, args=(start_time, keys_checked, lock))
        tracker_process.start()

        check_key_partial = partial(check_key, num_workers=NUM_WORKERS, keys_checked=keys_checked, lock=lock)

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(check_key_partial, i): i for i in range(NUM_WORKERS)}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    break

        stop_event.set()
        tracker_process.terminate()
        tracker_process.join()

    end_time = time.time()
    if not key_found.value:
        print("Key not found")

    print(f"⏳ Full time: {end_time - start_time:.2f} sec.")
    os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()