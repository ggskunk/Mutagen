#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>
#include <unordered_map>
#include <cmath>
#include <openssl/ec.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <openssl/evp.h>

using namespace std;

// Configuration
const string TARGET_HASH160 = "29a78213caa9eea824acf08022ab9dfc83414f56";
const string BASE_KEY = "00000000000000000000000000000000000000000000000000000000000d2c55";
const int PUZZLE_NUM = 21;
const int WORKERS = thread::hardware_concurrency();
const size_t REPORT_INTERVAL = 10000;

#ifndef NID_secp256k1
#define NID_secp256k1 714
#endif

// Historical flip counts
const unordered_map<int, int> FLIP_TABLE = {
    {20, 8}, {21, 9}, {22, 11}, {23, 12}, {24, 9}, {25, 12}, {26, 14}, {27, 13},
    {28, 16}, {29, 18}, {30, 16}, {31, 13}, {32, 14}, {33, 15}, {34, 16}, {35, 19},
    {36, 14}, {37, 23}, {38, 21}, {39, 23}, {40, 20}, {41, 25}, {42, 24}, {43, 19},
    {44, 24}, {45, 21}, {46, 24}, {47, 27}, {48, 21}, {49, 30}, {50, 29}, {51, 25},
    {52, 27}, {53, 26}, {54, 30}, {55, 31}, {56, 31}, {57, 33}, {58, 28}, {59, 30},
    {60, 31}, {61, 25}, {62, 35}, {63, 34}, {64, 34}, {65, 37}, {66, 35}, {67, 31},
    {68, 34}
};

// Global variables
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, size_t, int>> results;

// Predict flip count
int predict_flips(int puzzle_num) {
    if (FLIP_TABLE.count(puzzle_num)) {
        return FLIP_TABLE.at(puzzle_num);
    }
    return 8; // Default
}

// Binomial coefficient calculation
size_t combinations_count(int n, int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    size_t result = n;
    for(int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

// Memory-efficient combination generator
class CombinationGenerator {
    int n, k;
    vector<int> current;
public:
    CombinationGenerator(int n, int k) : n(n), k(k), current(k) {
        for (int i = 0; i < k; ++i) current[i] = i;
    }
   
    bool next() {
        int i = k - 1;
        while (i >= 0 && current[i] == n - k + i) --i;
        if (i < 0) return false;
       
        ++current[i];
        for (int j = i + 1; j < k; ++j)
            current[j] = current[j-1] + 1;
        return true;
    }
   
    const vector<int>& get() const { return current; }
};

// Fast BIGNUM XOR
void bn_xor(BIGNUM* r, const BIGNUM* a, const BIGNUM* b) {
    BN_copy(r, a);
    for (int i = 0; i < max(BN_num_bits(a), BN_num_bits(b)); ++i) {
        if (BN_is_bit_set(a, i) != BN_is_bit_set(b, i))
            BN_set_bit(r, i);
        else
            BN_clear_bit(r, i);
    }
}

// Compute HASH160
string compute_hash160(const BIGNUM* priv_key) {
    EC_KEY* key = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (!key) return "";
   
    if (!EC_KEY_set_private_key(key, priv_key)) {
        EC_KEY_free(key);
        return "";
    }
   
    const EC_GROUP* group = EC_KEY_get0_group(key);
    EC_POINT* pub = EC_POINT_new(group);
    if (!EC_POINT_mul(group, pub, priv_key, nullptr, nullptr, nullptr)) {
        EC_POINT_free(pub);
        EC_KEY_free(key);
        return "";
    }
   
    unsigned char pubkey[33];
    if (EC_POINT_point2oct(group, pub, POINT_CONVERSION_COMPRESSED, pubkey, 33, nullptr) != 33) {
        EC_POINT_free(pub);
        EC_KEY_free(key);
        return "";
    }
    EC_POINT_free(pub);
   
    unsigned char sha256[SHA256_DIGEST_LENGTH];
    SHA256(pubkey, 33, sha256);
   
    unsigned char ripemd160[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha256, SHA256_DIGEST_LENGTH, ripemd160);
   
    char hex[41];
    for (int i = 0; i < 20; i++)
        snprintf(hex + 2 * i, sizeof(hex) - 2 * i, "%02x", ripemd160[i]);
    return string(hex);
}

// Worker function
void worker(BIGNUM* base_bn, int bit_length, int flip_count, size_t start, size_t end) {
    CombinationGenerator gen(bit_length, flip_count);
    for (size_t i = 0; i < start && gen.next(); ++i);
   
    BIGNUM* current = BN_new();
    BIGNUM* mask = BN_new();
    size_t count = 0;
    size_t last_report = 0;
   
    do {
        BN_zero(mask);
        for (int pos : gen.get()) BN_set_bit(mask, pos);
       
        bn_xor(current, base_bn, mask);
        string hash160 = compute_hash160(current);
       
        if (hash160 == TARGET_HASH160) {
            char* hex = BN_bn2hex(current);
            lock_guard<mutex> lock(result_mutex);
            results.push(make_tuple(hex, start + count + 1, flip_count));
            OPENSSL_free(hex);
            stop_event.store(true);
            break;
        }
       
        if (++count - last_report >= REPORT_INTERVAL) {
            cout << "Worker progress: " << count << " combinations ("
                 << fixed << setprecision(2) << (count*100.0/(end-start)) << "%)\r";
            last_report = count;
        }
           
    } while (gen.next() && (count < (end-start)) && !stop_event.load());
   
    BN_free(current);
    BN_free(mask);
}

int main() {
    cout << "=======================================\n";
    cout << "== Mutagen Puzzle Solver by Denevron ==\n";
    cout << "=======================================\n";
   
    BIGNUM* base_bn = BN_new();
    BN_hex2bn(&base_bn, BASE_KEY.c_str());
   
    const int flip_count = predict_flips(PUZZLE_NUM);   // Количество изменяемых бит для головоломки
    const int bit_length = PUZZLE_NUM;                    // Длина головоломки, например 20 бит для головоломки 20
    const size_t total_combs = combinations_count(bit_length, flip_count);  // Рассчитываем количество комбинаций для данной головоломки
    
    cout << "Searching Puzzle " << PUZZLE_NUM << " (256-bit)\n";
    cout << "Base Key: " << BASE_KEY << "\n";
    cout << "Target HASH160: " << TARGET_HASH160 << "\n";
    cout << "Predicted Flip Count: " << flip_count << " bits (from FLIP_TABLE)\n";
    cout << "Total Possible Combinations: " << total_combs << "\n";
    cout << "Using " << WORKERS << " workers...\n";
   
    auto start_time = chrono::high_resolution_clock::now();
    vector<thread> threads;
    size_t chunk = total_combs / WORKERS;
   
    for (int i = 0; i < WORKERS; ++i) {
        size_t start = i * chunk;
        size_t end = (i == WORKERS-1) ? total_combs : start + chunk;
        threads.emplace_back(worker, base_bn, bit_length, flip_count, start, end);
    }
   
    for (auto& t : threads) t.join();
    BN_free(base_bn);
   
    // Защита от ошибок при отсутствии решений
    if (!results.empty()) {
        auto [hex_key, checked, flips] = results.front();
        auto elapsed = chrono::duration_cast<chrono::seconds>(
            chrono::high_resolution_clock::now() - start_time).count();

        // Защита от деления на ноль при вычислении скорости
        double speed = (elapsed > 0) ? (checked / elapsed) : 0.0;
       
        cout << "\n\n== SOLUTION FOUND ==\n";
        cout << "Private Key: " << hex_key << "\n";
        cout << "Actual Bit Flips: " << flips << "\n";
        cout << "Keys Checked: " << checked << "\n";
        cout << "Search Time: " << elapsed << " seconds\n";
        cout << "Speed: " << fixed << setprecision(2) << speed << " keys/sec\n";
       
        ofstream out("puzzle_" + to_string(PUZZLE_NUM) + "_solution.txt");
        out << hex_key;
        out.close();
        cout << "Solution saved to puzzle_" << PUZZLE_NUM << "_solution.txt\n";
    } else {
        cout << "\nSolution not found. Try adjusting flip count ±2\n";
    }
   
    return 0;
}
