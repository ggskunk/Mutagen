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
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <openssl/ec.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <openssl/evp.h>

using namespace std;

// Configuration defaults
int PUZZLE_NUM = 20;
int WORKERS = thread::hardware_concurrency();
const size_t REPORT_INTERVAL = 10000;
const size_t BATCH_SIZE = 1000;
atomic<size_t> current_index(0);

// Historical puzzle data (puzzle number: {flip count, target hash, private key decimal})
const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {
    {20, {8, "b907c3a2a3b27789dfb509b730dd47703c272868", "357535"}}, 
    {21, {9, "29a78213caa9eea824acf08022ab9dfc83414f56", "863317"}},
    {22, {11, "7ff45303774ef7a52fffd8011981034b258cb86b", "1811764"}}, 
    {23, {12, "d0a79df189fe1ad5c306cc70497b358415da579e", "3007503"}},
    {24, {9, "0959e80121f36aea13b3bad361c15dac26189e2f", "5598802"}},
    {25, {12, "2f396b29b27324300d0c59b17c3abc1835bd3dbb", "14428676"}},
    {26, {14, "bfebb73562d4541b32a02ba664d140b5a574792f", "33185509"}},
    {27, {13, "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560", "54538862"}},
    {28, {16, "1306b9e4ff56513a476841bac7ba48d69516b1da", "111949941"}},
    {29, {18, "5a416cc9148f4a377b672c8ae5d3287adaafadec", "227634408"}},
    {30, {16, "d39c4704664e1deb76c9331e637564c257d68a08", "400708894"}},
    {31, {13, "d805f6f251f7479ebd853b3d0f4b9b2656d92f1d", "1033162084"}},
    {32, {14, "9e42601eeaedc244e15f17375adb0e2cd08efdc9", "2102388551"}},
    {33, {15, "4e15e5189752d1eaf444dfd6bff399feb0443977", "3093472814"}},
    {34, {16, "f6d67d7983bf70450f295c9cb828daab265f1bfa", "7137437912"}},
    {35, {19, "f6d8ce225ffbdecec170f8298c3fc28ae686df25", "14133072157"}},
    {36, {14, "74b1e012be1521e5d8d75e745a26ced845ea3d37", "20112871792"}},
    {37, {23, "28c30fb11ed1da72e7c4f89c0164756e8a021d", "42387769980"}},
    {38, {21, "b190e2d40cfdeee2cee072954a2be89e7ba39364", "100251560595"}},
    {39, {23, "0b304f2a79a027270276533fe1ed4eff30910876", "146971536592"}},
    {40, {20, "95a156cd21b4a69de969eb6716864f4c8b82a82a", "323724968937"}},
    {41, {25, "d1562eb37357f9e6fc41cb2359f4d3eda4032329", "1003651412950"}},
    {42, {24, "8efb85f9c5b5db2d55973a04128dc7510075ae23", "1458252205147"}},
    {43, {19, "f92044c7924e5525c61207972c253c9fc9f086f7", "2895374552463"}},
    {44, {24, "80df54e1f612f2fc5bdc05c9d21a83aa8d20791e", "7409811047825"}},
    {45, {21, "f0225bfc68a6e17e87cd8b5e60ae3be18f120753", "15404761757071"}},
    {46, {24, "9a012260d01c5113df66c8a8438c9f7a1e3d5dac", "19996463086597"}},
    {47, {27, "f828005d41b0f4fed4c8dca3b06011072cfb07d4", "51408670348612"}},
    {48, {21, "8661cb56d9df0a61f01328b55af7e56a3fe7a2b2", "119666659114170"}},
    {49, {30, "0d2f533966c6578e1111978ca698f8add7fffdf3", "191206974700443"}},
    {50, {29, "de081b76f840e462fa2cdf360173dfaf4a976a47", "409118905032525"}},
    {51, {25, "ef6419cffd7fad7027994354eb8efae223c2dbe7", "611140496167764"}},
    {52, {27, "36af659edbe94453f6344e920d143f1778653ae7", "2058769515153876"}},
    {53, {26, "2f4870ef54fa4b048c1365d42594cc7d3d269551", "4216495639600700"}},
    {54, {30, "cb66763cf7fde659869ae7f06884d9a0f879a092", "6763683971478124"}},
    {55, {31, "db53d9bbd1f3a83b094eeca7dd970bd85b492fa2", "9974455244496707"}},
    {56, {31, "48214c5969ae9f43f75070cea1e2cb41d5bdcccd", "30045390491869460"}},
    {57, {33, "328660ef43f66abe2653fa178452a5dfc594c2a1", "44218742292676575"}},
    {58, {28, "8c2a6071f89c90c4dab5ab295d7729d1b54ea60f", "138245758910846492"}},
    {59, {30, "b14ed3146f5b2c9bde1703deae9ef33af8110210", "199976667976342049"}},
    {60, {31, "cdf8e5c7503a9d22642e3ecfc87817672787b9c5", "525070384258266191"}},
    {61, {25, "68133e19b2dfb9034edf9830a200cfdf38c90cbd", "1135041350219496382"}},
    {62, {35, "e26646db84b0602f32b34b5a62ca3cae1f91b779", "1425787542618654982"}},
    {63, {34, "ef58afb697b094423ce90721fbb19a359ef7c50e", "3908372542507822062"}},
    {64, {34, "3ee4133d991f52fdf6a25c9834e0745ac74248a4", "8993229949524469768"}},
    {65, {37, "52e763a7ddc1aa4fa811578c491c1bc7fd570137", "17799667357578236628"}},
    {66, {35, "20d45a6a762535700ce9e0b216e31994335db8a5", "30568377312064202855"}},
    {67, {31, "739437bb3dd6d1983e66629c5f08c70e52769371", "46346217550346335726"}},
    {68, {34, "e0b8a2baee1b77fc703455f39d51477451fc8cfc", "132656943602386256302"}} 
};

// Global variables
vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
string BASE_KEY;
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, size_t, int>> results;
atomic<size_t> total_checked(0);
size_t total_combinations = 0;

static void printUsage(const char *programName) {
    cerr << "Usage: " << programName << " [-p <PUZZLE_NUM>] [-t <WORKERS>] [-h]\n";
    cerr << "Options:\n";
    cerr << "  -p <num>   Puzzle number to solve (default: 20)\n";
    cerr << "  -t <num>   Number of CPU threads to use (default: all cores)\n";
    cerr << "  -h         Show this help message\n";
    cerr << "\nAvailable puzzles: ";
    for (const auto& entry : PUZZLE_DATA) {
        cerr << entry.first << " ";
    }
    cerr << "\n";
}

string decimal_to_hex64(const string& decimal_str) {
    BIGNUM* bn = BN_new();
    BN_dec2bn(&bn, decimal_str.c_str());
    
    char* hex = BN_bn2hex(bn);
    string hex_str(hex);
    OPENSSL_free(hex);
    
    if (hex_str.length() < 64) {
        hex_str = string(64 - hex_str.length(), '0') + hex_str;
    }
    
    BN_free(bn);
    return hex_str;
}

vector<unsigned char> hex_to_bytes(const string& hex) {
    vector<unsigned char> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        string byteString = hex.substr(i, 2);
        unsigned char byte = static_cast<unsigned char>(strtoul(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

BIGNUM* get_last_n_bits(BIGNUM* num, int n) {
    BIGNUM* result = BN_new();
    BN_zero(result);
    
    for (int i = 0; i < n; ++i) {
        if (BN_is_bit_set(num, i)) {
            BN_set_bit(result, i);
        }
    }
    return result;
}

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

class CombinationGenerator {
    int n, k;
    vector<int> current;
    
public:
    CombinationGenerator(int n, int k) : n(n), k(k), current(k) {
        for (int i = 0; i < k; ++i) current[i] = i;
    }
  
    const vector<int>& get() const { return current; }
  
    bool next() {
        int i = k - 1;
        while (i >= 0 && current[i] == n - k + i) --i;
        if (i < 0) return false;
      
        ++current[i];
        for (int j = i + 1; j < k; ++j)
            current[j] = current[j-1] + 1;
        return true;
    }
  
    void unrank(size_t rank) {
        size_t remaining_rank = rank;
        int a = n;
        int b = k;
        size_t x = (combinations_count(n, k) - 1) - rank;
        
        for (int i = 0; i < k; i++) {
            a = largest_a_where_comb_a_b_le_x(a, b, x);
            current[i] = (n - 1) - a;
            x -= combinations_count(a, b);
            b--;
        }
    }
  
private:
    int largest_a_where_comb_a_b_le_x(int a, int b, size_t x) const {
        while (a >= b && combinations_count(a, b) > x) {
            a--;
        }
        return a;
    }
};

void bn_xor_nbits(BIGNUM* r, const BIGNUM* a, const BIGNUM* b, int n) {
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* mask = BN_new();
    BN_one(mask);
    BN_lshift(mask, mask, n);
    BN_sub_word(mask, 1);

    BIGNUM* tmp = BN_new();
    BN_copy(tmp, a);
    
    for (int i = 0; i < n; i++) {
        int a_bit = BN_is_bit_set(a, i);
        int b_bit = BN_is_bit_set(b, i);
        if (a_bit != b_bit) {
            BN_set_bit(tmp, i);
        } else {
            BN_clear_bit(tmp, i);
        }
    }

    BN_mod(r, tmp, mask, ctx);

    BN_free(tmp);
    BN_free(mask);
    BN_CTX_free(ctx);
}

void worker(BIGNUM* base_bn, int bit_length, int flip_count) {
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* padded_base = BN_new();
    BIGNUM* trimmed_base = nullptr;
    BIGNUM* current = BN_new();
    BIGNUM* mask = BN_new();
    EC_KEY* key = EC_KEY_new_by_curve_name(NID_secp256k1);
    EC_GROUP* group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT* pub = EC_POINT_new(group);

    // Convert base_bn to padded hex
    char* base_hex = BN_bn2hex(base_bn);
    string base_key_str(base_hex);
    OPENSSL_free(base_hex);
    
    if (base_key_str.length() < 64) {
        base_key_str = string(64 - base_key_str.length(), '0') + base_key_str;
    }
    
    BN_hex2bn(&padded_base, base_key_str.c_str());
    trimmed_base = get_last_n_bits(padded_base, bit_length);

    // Initialize digest contexts
    EVP_MD_CTX* sha256_ctx = EVP_MD_CTX_new();
    EVP_MD_CTX* ripemd160_ctx = EVP_MD_CTX_new();
    unsigned char pubkey[33];
    unsigned char sha256[SHA256_DIGEST_LENGTH];
    unsigned char ripemd160[RIPEMD160_DIGEST_LENGTH];

    while (!stop_event.load()) {
        size_t start = current_index.fetch_add(BATCH_SIZE);
        if (start >= total_combinations) break;
        
        CombinationGenerator gen(bit_length, flip_count);
        gen.unrank(start);
        
        size_t count = 0;
        while (count < BATCH_SIZE && start + count < total_combinations && !stop_event.load()) {
            BN_zero(mask);
            for (int pos : gen.get()) {
                BN_set_bit(mask, pos);
            }

            bn_xor_nbits(current, trimmed_base, mask, bit_length);

            if (!EC_KEY_set_private_key(key, current)) continue;
            if (!EC_POINT_mul(group, pub, current, nullptr, nullptr, ctx)) continue;
            if (EC_POINT_point2oct(group, pub, POINT_CONVERSION_COMPRESSED, pubkey, 33, ctx) != 33) continue;

            // Hash operations
            EVP_DigestInit_ex(sha256_ctx, EVP_sha256(), nullptr);
            EVP_DigestUpdate(sha256_ctx, pubkey, 33);
            EVP_DigestFinal_ex(sha256_ctx, sha256, nullptr);
            
            EVP_DigestInit_ex(ripemd160_ctx, EVP_ripemd160(), nullptr);
            EVP_DigestUpdate(ripemd160_ctx, sha256, SHA256_DIGEST_LENGTH);
            EVP_DigestFinal_ex(ripemd160_ctx, ripemd160, nullptr);

            if (memcmp(ripemd160, TARGET_HASH160_RAW.data(), 20) == 0) {
                char* hex_key = BN_bn2hex(current);
                lock_guard<mutex> lock(result_mutex);
                results.push(make_tuple(hex_key, total_checked.load(), flip_count));
                OPENSSL_free(hex_key);
                stop_event.store(true);
                break;
            }

            if (++total_checked % REPORT_INTERVAL == 0) {
                double progress = (double)total_checked / total_combinations * 100;
                cout << "Progress: " << fixed << setprecision(6) << progress << "% (";
                cout << total_checked << "/" << total_combinations << ")\r";
                cout.flush();
            }
            
            if (!gen.next()) break;
            count++;
        }
    }

    // Cleanup
    EVP_MD_CTX_free(sha256_ctx);
    EVP_MD_CTX_free(ripemd160_ctx);
    EC_POINT_free(pub);
    EC_GROUP_free(group);
    EC_KEY_free(key);
    BN_free(mask);
    BN_free(current);
    BN_free(trimmed_base);
    BN_free(padded_base);
    BN_CTX_free(ctx);
}

void initialize_openssl() {
    OPENSSL_init_crypto(OPENSSL_INIT_LOAD_CRYPTO_STRINGS | OPENSSL_INIT_ADD_ALL_CIPHERS | OPENSSL_INIT_ADD_ALL_DIGESTS, nullptr);
}

int main(int argc, char* argv[]) {
    // Get puzzle data
    auto puzzle_it = PUZZLE_DATA.find(PUZZLE_NUM);
    if (puzzle_it == PUZZLE_DATA.end()) {
        cerr << "Error: Invalid puzzle number\n";
        return 1;
    }
    auto [FLIP_COUNT, TARGET_HASH160, PRIVATE_KEY_DECIMAL] = puzzle_it->second;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-p") {
            if (i + 1 < argc) {
                PUZZLE_NUM = stoi(argv[++i]);
                puzzle_it = PUZZLE_DATA.find(PUZZLE_NUM);
                if (puzzle_it == PUZZLE_DATA.end()) {
                    cerr << "Error: Invalid puzzle number. Available puzzles: ";
                    for (const auto& entry : PUZZLE_DATA) {
                        cerr << entry.first << " ";
                    }
                    cerr << "\n";
                    return 1;
                }
                tie(FLIP_COUNT, TARGET_HASH160, PRIVATE_KEY_DECIMAL) = puzzle_it->second;
            } else {
                cerr << "Error: -p requires a puzzle number argument\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-t") {
            if (i + 1 < argc) {
                WORKERS = stoi(argv[++i]);
                if (WORKERS <= 0) {
                    cerr << "Error: Number of workers must be positive\n";
                    return 1;
                }
            } else {
                cerr << "Error: -t requires a thread count argument\n";
                printUsage(argv[0]);
                return 1;
            }
        } else {
            cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    initialize_openssl();

    BASE_KEY = decimal_to_hex64(PRIVATE_KEY_DECIMAL);
    TARGET_HASH160_RAW = hex_to_bytes(TARGET_HASH160);
    const int bit_length = PUZZLE_NUM;
    total_combinations = combinations_count(bit_length, FLIP_COUNT);

    cout << "=======================================\n";
    cout << "== Mutagen Puzzle Solver by Denevron ==\n";
    cout << "=======================================\n";
    cout << "Searching Puzzle " << PUZZLE_NUM << " (" << bit_length << "-bit)\n";
    cout << "Base Key: " << BASE_KEY.substr(0, 10) << "..." << BASE_KEY.substr(BASE_KEY.length()-10) << "\n";
    cout << "Target HASH160: " << TARGET_HASH160.substr(0, 10) << "..." << TARGET_HASH160.substr(TARGET_HASH160.length()-10) << "\n";
    cout << "Predicted Flip Count: " << FLIP_COUNT << " bits\n";
    cout << "Total Combinations: " << total_combinations << "\n";
    cout << "Using " << WORKERS << " workers...\n";

    BIGNUM* base_bn = BN_new();
    BN_hex2bn(&base_bn, BASE_KEY.c_str());

    auto start_time = chrono::high_resolution_clock::now();
    vector<thread> threads;
    
    for (int i = 0; i < WORKERS; ++i) {
        threads.emplace_back(worker, base_bn, bit_length, FLIP_COUNT);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    BN_free(base_bn);

    if (!results.empty()) {
        auto [hex_key, checked, flips] = results.front();
        auto elapsed = chrono::duration_cast<chrono::seconds>(
            chrono::high_resolution_clock::now() - start_time).count();
      
        cout << "\n=======================================\n";
        cout << "=========== SOLUTION FOUND ============\n";
        cout << "=======================================\n";
        cout << "Private Key: " << hex_key << "\n";
        cout << "Search Time: " << elapsed << " seconds\n";
        cout << "Keys Checked: " << checked << "\n";
        cout << "Bit Flips: " << flips << endl;
      
        ofstream out("puzzle_" + to_string(PUZZLE_NUM) + "_solution.txt");
        out << hex_key;
        out.close();
        cout << "Solution saved to puzzle_" << PUZZLE_NUM << "_solution.txt\n";
    } else {
        cout << "\nSolution not found. Checked " << total_checked << " combinations\n";
    }

    return 0;
}
