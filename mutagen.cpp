#include <iostream>
#include <array>
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
#include <immintrin.h>
#include <omp.h>
#include <csignal>
#include <random>
#include <algorithm>
#include <getopt.h>

#ifdef _WIN32
    #include <windows.h>
#endif

// Include the required headers
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

using namespace std;

// Cross-platform terminal functions
void initConsole() {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

void clearTerminal() {
#ifdef _WIN32
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = {0, 0};
    DWORD count;
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hStdOut, &csbi);
    FillConsoleOutputCharacter(hStdOut, ' ', csbi.dwSize.X * csbi.dwSize.Y, coord, &count);
    SetConsoleCursorPosition(hStdOut, coord);
#else
    std::cout << "\033[2J\033[H";
#endif
    std::cout.flush();
}

void moveCursorTo(int x, int y) {
#ifdef _WIN32
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = {(SHORT)x, (SHORT)y};
    SetConsoleCursorPosition(hStdOut, coord);
#else
    std::cout << "\033[" << y << ";" << x << "H";
#endif
    std::cout.flush();
}

// Configuration defaults
int PUZZLE_NUM = -1; // Initialize to invalid to force user input via -p
int WORKERS = omp_get_num_procs();
int FLIP_COUNT = -1; // Will be set based on puzzle number unless overridden
const __uint128_t REPORT_INTERVAL = 10000000;
static constexpr int POINTS_BATCH_SIZE = 256;
static constexpr int HASH_BATCH_SIZE = 8;

// Historical puzzle data (puzzle number: {flip count, target hash, private key decimal})
const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {
    {20, {8, "b907c3a2a3b27789dfb509b730dd47703c272868",  "357535"}},
    {68, {34, "e0b8a2baee1b77fc703455f39d51477451fc8cfc", "132656943602386256302"}}
    // Add other puzzle data here if needed
};

// Global variables
vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
Int BASE_KEY; // This will be set based on puzzle or manual input
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, __uint128_t, int>> results; // Stores {hex_key, combination_index, flip_count}

// AVX2 128-bit counter implementation
union AVXCounter {
    __m256i vec;
    uint64_t u64[4];
    __uint128_t u128[2];

    AVXCounter() : vec(_mm256_setzero_si256()) {}

    AVXCounter(__uint128_t value) {
        store(value);
    }

    void increment() {
        __m256i one = _mm256_set_epi64x(0, 0, 0, 1);
        vec = _mm256_add_epi64(vec, one);
        if (u64[0] == 0) { // Check low 64 bits for carry
            __m256i carry = _mm256_set_epi64x(0, 0, 1, 0);
            vec = _mm256_add_epi64(vec, carry);
        }
    }

    void add(__uint128_t value) {
        __uint128_t low_add = value & 0xFFFFFFFFFFFFFFFFULL;
        __uint128_t high_add = value >> 64;
        __m256i add_val = _mm256_set_epi64x(0, 0, high_add, low_add);
        __uint128_t old_low = u64[0]; // Store old low value before add
        vec = _mm256_add_epi64(vec, add_val);
        if (u64[0] < old_low) { // Check for carry out of low 64 bits
            __m256i carry = _mm256_set_epi64x(0, 0, 1, 0);
            vec = _mm256_add_epi64(vec, carry);
        }
    }

    __uint128_t load() const {
        return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0];
    }

    void store(__uint128_t value) {
        u64[0] = static_cast<uint64_t>(value);
        u64[1] = static_cast<uint64_t>(value >> 64);
        u64[2] = 0;
        u64[3] = 0;
    }

    bool operator<(const AVXCounter& other) const {
        if (u64[1] != other.u64[1])
            return u64[1] < other.u64[1];
        return u64[0] < other.u64[0];
    }

    bool operator>=(const AVXCounter& other) const {
        return !(*this < other);
    }

    static AVXCounter div(const AVXCounter& num, uint64_t denom) {
        __uint128_t n = num.load();
        __uint128_t q = n / denom;
        return AVXCounter(q);
    }

    static uint64_t mod(const AVXCounter& num, uint64_t denom) {
        __uint128_t n = num.load();
        return n % denom;
    }

    static AVXCounter mul(uint64_t a, uint64_t b) {
        __uint128_t result = static_cast<__uint128_t>(a) * b;
        return AVXCounter(result);
    }
};

// Global counters and performance variables
__uint128_t total_combinations = 0;
atomic<uint64_t> globalComparedCount(0); // Total keys checked across all threads
chrono::time_point<chrono::high_resolution_clock> tStart;
double globalElapsedTime = 0.0;
double mkeysPerSec = 0.0;
mutex progress_mutex; // Mutex for updating the terminal progress display

// Helper functions
static std::string formatElapsedTime(double seconds) {
    int hrs = static_cast<int>(seconds) / 3600;
    int mins = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

static std::string to_string_128(__uint128_t value) {
    if (value == 0) return "0";
    std::string str;
    while (value > 0) {
        str += char('0' + (value % 10));
        value /= 10;
    }
    std::reverse(str.begin(), str.end());
    return str.empty() ? "0" : str;
}

// Signal handler for clean shutdown
void signalHandler(int signum) {
    stop_event.store(true);
    cout << "\nInterrupt received (" << signum << "), shutting down gracefully...\n";
}

// Combination generator class
class CombinationGenerator {
    int n, k;
    std::vector<int> current;
public:
    CombinationGenerator(int n, int k) : n(n), k(k), current(k) {
        if (k > n || k < 0) k = 0; // Handle invalid k
        this->k = k; // Store potentially adjusted k
        this->n = n; // Store n
        if (k > 0) { // Initialize only if k > 0
             for (int i = 0; i < k; ++i) current[i] = i;
        } else {
             current.clear(); // Ensure current is empty for k=0
        }
    }

    static __uint128_t combinations_count(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = n - k;

        __uint128_t result = 1;
        for(int i = 1; i <= k; ++i) {
            // Check potential overflow before multiplication
            unsigned __int128 temp_res;
            if (__builtin_mul_overflow(result, (unsigned __int128)(n - i + 1), &temp_res)) {
                cerr << "\nWarning: Combination count overflow detected (n=" << n << ", k=" << i << ").\n";
                return -1; // Indicate overflow error
            }
            result = temp_res;

            // Check potential division by zero (shouldn't happen if k >= 1)
            if (i == 0) {
                cerr << "\nError: Division by zero in combinations_count.\n";
                return -1; // Indicate error
            }
            result /= i;
        }
        return result;
    }

    const std::vector<int>& get() const { return current; }

    bool next() {
        if (k <= 0 || k > n) return false; // No next if k is invalid or 0
        int i = k - 1;
        while (i >= 0 && current[i] == n - k + i) --i;
        if (i < 0) return false; // Last combination reached

        ++current[i];
        for (int j = i + 1; j < k; ++j)
            current[j] = current[j - 1] + 1;
        return true;
    }

    void unrank(__uint128_t rank) {
        if (k <= 0 || k > n) {
            current.clear();
            return; // Handle invalid k
        }
        __uint128_t total = combinations_count(n, k);
         if (rank >= total && total != static_cast<__uint128_t>(-1)) { // Check rank validity, ignore if total overflowed
            cerr << "\nWarning: Rank " << to_string_128(rank) << " is out of bounds (total combinations: " << to_string_128(total) << "). Cannot unrank.\n";
             current.clear(); // Or set to a default state?
             // Setting to first combination might be an option, but clear indicates failure.
             // for (int i = 0; i < k; ++i) current[i] = i; // Reset to first? No, clear is better.
             return;
        }

        current.resize(k);
        __uint128_t remaining_rank = rank;
        int current_n = n;
        int current_k = k;
        int last_pivot = -1; // Keep track of the last chosen element's index relative to original n

        for (int i = 0; i < k; ++i) {
            if (current_k <= 0 || current_n < current_k) {
                 // Should not happen if rank is valid, but good for safety
                 cerr << "\nError: Invalid state in unrank (n=" << current_n << ", k=" << current_k << ").\n";
                 current.clear();
                 return;
            }

            int pivot = 0; // Element index relative to current_n
            while (true) {
                if (current_n - pivot < current_k) { // Check if enough elements remain
                     cerr << "\nError: Logic error during unrank pivot search (n=" << current_n << ", k=" << current_k << ", pivot=" << pivot <<"). Rank was " << to_string_128(rank) << "\n";
                     current.clear();
                     return; // Cannot form combination, indicates error
                }
                 __uint128_t comb = combinations_count(current_n - pivot - 1, current_k - 1);
                 if (comb == static_cast<__uint128_t>(-1)) { // Handle combination calculation overflow during unrank
                     cerr << "\nError: Combination count overflow during unrank.\n";
                     current.clear();
                     return;
                 }
                 if (remaining_rank < comb) break; // Found the correct pivot range

                 remaining_rank -= comb;
                 pivot++;
            }
            current[i] = pivot + last_pivot + 1; // Calculate absolute index
            last_pivot = current[i]; // Update last absolute index
            current_n -= (pivot + 1); // Reduce n for the next iteration
            current_k--; // Reduce k
        }
    }
};

// Hashing helper functions
inline void prepareShaBlock(const uint8_t* dataSrc, size_t dataLen, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, dataLen);
    outBlock[dataLen] = 0x80;
    uint64_t bitLen = static_cast<uint64_t>(dataLen) * 8;
    // Place the 64-bit length in big-endian order (standard for SHA)
    for(int i = 0; i < 8; ++i) {
        outBlock[63 - i] = (uint8_t)(bitLen >> (i * 8));
    }
    // Note: Assumes dataLen <= 55, no need for extra padding block for pubkeys (33 bytes)
}

inline void prepareRipemdBlock(const uint8_t* dataSrc32Bytes, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc32Bytes, 32); // RIPEMD input is SHA256 output (32 bytes)
    outBlock[32] = 0x80;
    const uint64_t bitLen = 256; // SHA256 output is always 256 bits
    // Place the 64-bit length in little-endian order (standard for RIPEMD)
    for(int i = 0; i < 8; ++i) {
        outBlock[56 + i] = (uint8_t)(bitLen >> (i * 8));
    }
}

// Optimized HASH160 function using AVX2 batching
static void computeHash160BatchBinSingle(int numKeys,
                                       uint8_t pubKeys[][33],
                                       uint8_t hashResults[][20])
{
    if (numKeys <= 0) return; // Nothing to do

    alignas(32) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
    alignas(32) std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
    alignas(32) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
    alignas(32) std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE> ripemdOutputs;

    // Pre-calculate padding blocks if needed (only once)
    static std::array<uint8_t, 64> shaPaddingBlock;
    static std::array<uint8_t, 64> ripemdPaddingBlock;
    static bool paddingInitialized = false;
    if (!paddingInitialized) {
         // Use a dummy pubkey structure for padding calculation
         uint8_t dummyPubkey[33] = {0x02}; // Minimal valid structure
         prepareShaBlock(dummyPubkey, 33, shaPaddingBlock.data());
         uint8_t dummyShaOut[32] = {0}; // Dummy SHA output
         prepareRipemdBlock(dummyShaOut, ripemdPaddingBlock.data());
         paddingInitialized = true;
    }


    const int numBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;

    for (int batch = 0; batch < numBatches; batch++) {
        const int batchStart = batch * HASH_BATCH_SIZE;
        const int batchCount = std::min(HASH_BATCH_SIZE, numKeys - batchStart);

        // Prepare SHA-256 input blocks
        for (int i = 0; i < batchCount; i++) {
            prepareShaBlock(pubKeys[batchStart + i], 33, shaInputs[i].data());
        }
        // Pad remaining slots if the last batch is not full
        for (int i = batchCount; i < HASH_BATCH_SIZE; i++) {
             std::memcpy(shaInputs[i].data(), shaPaddingBlock.data(), 64);
        }

        const uint8_t* inPtr[HASH_BATCH_SIZE];
        uint8_t* outPtrSha[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = shaInputs[i].data();
            outPtrSha[i] = shaOutputs[i].data();
        }

        // Perform SHA256 for the batch
        sha256avx2_8B(inPtr[0], inPtr[1], inPtr[2], inPtr[3],
                      inPtr[4], inPtr[5], inPtr[6], inPtr[7],
                      outPtrSha[0], outPtrSha[1], outPtrSha[2], outPtrSha[3],
                      outPtrSha[4], outPtrSha[5], outPtrSha[6], outPtrSha[7]);

        // Prepare RIPEMD-160 input blocks from SHA outputs
        for (int i = 0; i < batchCount; i++) {
            prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
        }
        // Pad remaining slots for RIPEMD if needed
         for (int i = batchCount; i < HASH_BATCH_SIZE; i++) {
             std::memcpy(ripemdInputs[i].data(), ripemdPaddingBlock.data(), 64);
         }

        uint8_t* outPtrRipemd[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = ripemdInputs[i].data(); // Input is now RIPEMD inputs
            outPtrRipemd[i] = ripemdOutputs[i].data();
        }

        // Perform RIPEMD160 for the batch
         ripemd160avx2::ripemd160avx2_32(
            (unsigned char*)inPtr[0], (unsigned char*)inPtr[1],
            (unsigned char*)inPtr[2], (unsigned char*)inPtr[3],
            (unsigned char*)inPtr[4], (unsigned char*)inPtr[5],
            (unsigned char*)inPtr[6], (unsigned char*)inPtr[7],
            outPtrRipemd[0], outPtrRipemd[1], outPtrRipemd[2], outPtrRipemd[3],
            outPtrRipemd[4], outPtrRipemd[5], outPtrRipemd[6], outPtrRipemd[7]
        );

        // Copy results for the valid keys in the batch
        for (int i = 0; i < batchCount; i++) {
            std::memcpy(hashResults[batchStart + i], ripemdOutputs[i].data(), 20);
        }
    }
}

// Worker thread function
void worker(Secp256K1* secp, int bit_length, int flip_count, int threadId, AVXCounter start_rank, AVXCounter end_rank) {
    const int fullBatchSize = 2 * POINTS_BATCH_SIZE; // Points generated per combination (plus and minus)
    alignas(32) uint8_t localPubKeys[HASH_BATCH_SIZE][33];
    alignas(32) uint8_t localHashResults[HASH_BATCH_SIZE][20];
    alignas(32) int pointIndices[HASH_BATCH_SIZE]; // To map hash result back to point index

    // Precompute target hash for faster comparison using SSE/AVX
    // Load only the first 16 bytes for the initial check
    __m128i target16_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(TARGET_HASH160_RAW.data()));

    // Precompute generator points [1*G, 2*G, ..., POINTS_BATCH_SIZE*G]
    alignas(32) Point plusPoints[POINTS_BATCH_SIZE];
    alignas(32) Point minusPoints[POINTS_BATCH_SIZE];
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
        Int tmp;
        tmp.SetInt32(i + 1); // Use offset i+1 to generate G, 2G, ...
        plusPoints[i] = secp->ComputePublicKey(&tmp);
        minusPoints[i] = plusPoints[i];
        minusPoints[i].y.ModNeg(); // Create the negative counterpart -[i+1]*G
    }

    // Storage for batch modular inverse and point addition
    alignas(32) Int deltaX[POINTS_BATCH_SIZE];
    IntGroup modGroup(POINTS_BATCH_SIZE);
    alignas(32) Int pointBatchX[fullBatchSize];
    alignas(32) Int pointBatchY[fullBatchSize];

    // Initialize combination generator for this thread's range
    CombinationGenerator gen(bit_length, flip_count);
    if (start_rank >= end_rank) return; // No work to do if range is empty
    gen.unrank(start_rank.load());
     // Check if unrank failed (returned empty vector)
     if (gen.get().empty() && flip_count > 0) { // Don't check if k=0
         cerr << "\nThread " << threadId << ": Unrank failed for start rank " << to_string_128(start_rank.load()) << ". Stopping thread.\n";
         return;
     }


    AVXCounter current_rank = start_rank;
    uint64_t thread_key_count = 0; // Keys checked by this thread since last global update

    while (!stop_event.load() && current_rank < end_rank) {
        Int currentKey;
        currentKey.Set(&BASE_KEY);
        const vector<int>& flips = gen.get();

        // If k=0, flips will be empty. Handle this case: only check BASE_KEY + offset*G
        if (flip_count == 0) {
             // The logic below implicitly handles k=0 because the BASE_KEY is used
             // and the loop `for (int pos : flips)` won't execute.
             // However, the `CombinationGenerator` part might need careful handling for k=0.
             // Let's assume the loop structure works correctly even for k=0 (generator provides one empty vector).
        } else if (flips.empty()) {
             // This might happen if unrank failed or k > n, should have been caught earlier.
              cerr << "\nThread " << threadId << ": Encountered empty flip vector unexpectedly for rank " << to_string_128(current_rank.load()) << ". Stopping.\n";
               break; // Stop processing for this thread
        }


        // Apply flips to the base key
        for (int pos : flips) {
            if (pos >= 0 && pos < bit_length) { // Safety check for bit position
                Int mask;
                mask.SetInt32(1);
                mask.ShiftL(pos);
                currentKey.Xor(&mask);
            } else {
                 cerr << "\nWarning: Invalid flip position " << pos << " encountered.\n";
                 // Decide how to handle: skip flip, log error, stop?
            }
        }

        // Compute the starting public key P = currentKey * G
        Point startPoint = secp->ComputePublicKey(&currentKey);
         if (startPoint.x.IsZero() && startPoint.y.IsZero()) {
            // If currentKey leads to point at infinity, skip this combination
            // (though highly unlikely for random-like keys)
            if (!gen.next()) break; // Advance to next combination or break if done
            current_rank.increment();
            continue;
        }
        Int startPointX, startPointY;
        startPointX.Set(&startPoint.x);
        startPointY.Set(&startPoint.y);

        // --- Batch Point Addition: Calculate P +/- [i+1]*G ---
        // Compute deltaX = plusPoints[i].x - startPoint.x for batch modular inverse
        for (int i = 0; i < POINTS_BATCH_SIZE; ++i) {
             deltaX[i].ModSub(&plusPoints[i].x, &startPointX);
        }
        // Batch modular inverse: deltaX becomes 1/deltaX mod p
        modGroup.Set(deltaX);
        modGroup.ModInv();

        // Compute resulting points using the inverted deltaX values
         for (int i = 0; i < POINTS_BATCH_SIZE; ++i) {
             // Check if deltaX was zero (points have same x-coordinate) before inverse
              // Note: ModInv() might handle this internally, check its implementation.
              // If ModInv sets result to 0 for zero input, the following calculations might be incorrect.
              // Assuming ModInv handles it or collision is rare enough to ignore for speed.

            // Process plus point i: R = P + [i+1]*G
            Int deltaY_plus; deltaY_plus.ModSub(&plusPoints[i].y, &startPointY);
            Int slope_plus; slope_plus.ModMulK1(&deltaY_plus, &deltaX[i]); // slope = (y2-y1)/(x2-x1)
            Int slopeSq_plus; slopeSq_plus.ModSquareK1(&slope_plus);

            pointBatchX[i].Set(&slopeSq_plus);
            pointBatchX[i].ModSub(&startPointX);
            pointBatchX[i].ModSub(&plusPoints[i].x); // Rx = s^2 - x1 - x2

            Int diffX_plus; diffX_plus.ModSub(&startPointX, &pointBatchX[i]); // x1 - Rx
            pointBatchY[i].ModMulK1(&slope_plus, &diffX_plus);
            pointBatchY[i].ModSub(&startPointY); // Ry = s*(x1-Rx) - y1

            // Process minus point i: R = P - [i+1]*G = P + ([i+1]*G).neg()
            Int deltaY_minus; deltaY_minus.ModSub(&minusPoints[i].y, &startPointY);
            Int slope_minus; slope_minus.ModMulK1(&deltaY_minus, &deltaX[i]); // slope = (y3-y1)/(x3-x1) where P3 = -P2
            Int slopeSq_minus; slopeSq_minus.ModSquareK1(&slope_minus);

            pointBatchX[POINTS_BATCH_SIZE + i].Set(&slopeSq_minus);
            pointBatchX[POINTS_BATCH_SIZE + i].ModSub(&startPointX);
            pointBatchX[POINTS_BATCH_SIZE + i].ModSub(&minusPoints[i].x); // Rx = s^2 - x1 - x3

            Int diffX_minus; diffX_minus.ModSub(&startPointX, &pointBatchX[POINTS_BATCH_SIZE + i]); // x1 - Rx
            pointBatchY[POINTS_BATCH_SIZE + i].ModMulK1(&slope_minus, &diffX_minus);
            pointBatchY[POINTS_BATCH_SIZE + i].ModSub(&startPointY); // Ry = s*(x1-Rx) - y1
         }

        // --- Hash and Check Batch ---
        int keysInCombination = 0; // Count keys processed for this combination rank
        for (int batchStart = 0; batchStart < fullBatchSize; batchStart += HASH_BATCH_SIZE) {
             int currentBatchSize = std::min(HASH_BATCH_SIZE, fullBatchSize - batchStart);
             int localBatchCount = 0; // Keys ready for hashing in this HASH_BATCH_SIZE batch

            // Prepare compressed public keys for the current HASH_BATCH_SIZE
             for (int i = 0; i < currentBatchSize; ++i) {
                  int pointIndex = batchStart + i; // Index in pointBatchX/Y (0 to fullBatchSize-1)

                  // Check if point is infinity (result of P +/- Q where P = +/- Q)
                  // This shouldn't happen if base points [i+1]*G are distinct from startPoint P
                  // and startPoint is not infinity. If it does, skip hashing.
                   if (pointBatchX[pointIndex].IsZero() && pointBatchY[pointIndex].IsZero()) {
                        continue; // Skip point at infinity
                   }

                   localPubKeys[localBatchCount][0] = pointBatchY[pointIndex].IsEven() ? 0x02 : 0x03;
                   for (int j = 0; j < 32; j++) {
                       localPubKeys[localBatchCount][32 - j] = pointBatchX[pointIndex].GetByte(j); // GetByte expects index 0..31
                   }
                   pointIndices[localBatchCount] = pointIndex; // Store original point index
                   localBatchCount++;
             }

            if (localBatchCount > 0) {
                // Compute HASH160 for the prepared keys
                computeHash160BatchBinSingle(localBatchCount, localPubKeys, localHashResults);
                thread_key_count += localBatchCount; // Update thread's local key counter
                keysInCombination += localBatchCount;

                // Check results against target
                for (int j = 0; j < localBatchCount; j++) {
                     // Fast check using SSE/AVX (first 16 bytes)
                     __m128i cand16_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(localHashResults[j]));
                     __m128i cmp_res = _mm_cmpeq_epi8(cand16_128, target16_128);
                     unsigned int mask = _mm_movemask_epi8(cmp_res);

                     if (mask == 0xFFFF) { // First 16 bytes match
                         // Full 20-byte comparison
                         bool fullMatch = true;
                         for (int k = 16; k < 20; k++) {
                             if (localHashResults[j][k] != TARGET_HASH160_RAW[k]) {
                                 fullMatch = false;
                                 break;
                             }
                         }

                        if (fullMatch) {
                            // --- SOLUTION FOUND ---
                            stop_event.store(true); // Signal other threads

                            // Calculate the final private key: k_final = k_base_flipped +/- offset
                            Int foundKey;
                            foundKey.Set(&currentKey); // Start with the flipped base key
                            int originalPointIndex = pointIndices[j]; // Index 0 to fullBatchSize-1
                            int offsetValue = (originalPointIndex % POINTS_BATCH_SIZE) + 1; // Offset is 1 to POINTS_BATCH_SIZE

                            Int offsetInt;
                            offsetInt.SetInt32(offsetValue);

                            if (originalPointIndex < POINTS_BATCH_SIZE) {
                                // Case: P + offset*G => k_final = k_current + offset
                                foundKey.Add(&offsetInt);
                            } else {
                                // Case: P - offset*G => k_final = k_current - offset
                                foundKey.Sub(&offsetInt);
                            }

                            string hexKey = foundKey.GetBase16();
                            hexKey = string(64 - min((int)hexKey.length(), 64), '0') + hexKey.substr(max(0, (int)hexKey.length() - 64)); // Pad to 64 hex chars

                            // --- Critical section for reporting result ---
                            lock_guard<mutex> lock(result_mutex);
                            if (results.empty()) { // Ensure only the first found result is stored
                                results.push(make_tuple(hexKey, current_rank.load(), flip_count));
                                // Update global count immediately upon finding result
                                globalComparedCount += thread_key_count;
                                thread_key_count = 0; // Reset thread count
                            }
                            // --- End Critical Section ---
                            return; // Exit worker function
                        }
                    } // End if mask == 0xFFFF
                } // End for loop checking hash results
            } // End if localBatchCount > 0

            // Check stop event periodically within the inner loop
            if (stop_event.load()) {
                 globalComparedCount += thread_key_count; // Update count before exiting
                 return;
            }
        } // End loop over HASH_BATCH_SIZE chunks

        // Update global count and report progress periodically
        // Calculate combinations checked by this thread
         __uint128_t combinations_done_thread = current_rank.load() - start_rank.load() + 1;

        if ((combinations_done_thread % (REPORT_INTERVAL / fullBatchSize + 1) == 0) || stop_event.load() || current_rank.load() == end_rank.load() - 1)
        {
             globalComparedCount += thread_key_count; // Add keys checked since last update
             thread_key_count = 0; // Reset counter

             if(threadId == 0 && !stop_event.load()) { // Let thread 0 update the display if not stopping
                 lock_guard<mutex> lock(progress_mutex);
                 auto now = chrono::high_resolution_clock::now();
                 globalElapsedTime = chrono::duration<double>(now - tStart).count();
                 uint64_t currentGlobalKeyCount = globalComparedCount.load(); // Read atomic counter

                 // Calculate overall progress based on estimated total keys or combinations
                 // Using combinations is more deterministic if total_combinations is accurate
                 double progressPercent = 0.0;
                 if (total_combinations > 0 && total_combinations != static_cast<__uint128_t>(-1)) {
                     // Estimate total checked combinations across all threads (approximate)
                      __uint128_t estimated_total_checked_comb = current_rank.load(); // Use rank of thread 0 as proxy
                      progressPercent = min(100.0, (double)estimated_total_checked_comb / total_combinations * 100.0);

                      // Better estimate? Sum progress of all threads? Maybe too complex.
                 }

                 if (globalElapsedTime > 0.1) {
                     mkeysPerSec = (double)currentGlobalKeyCount / globalElapsedTime / 1e6;
                 } else {
                     mkeysPerSec = 0.0;
                 }

                 moveCursorTo(0, 10); // Position cursor below header
                 cout << "Progress (Combinations): " << fixed << setprecision(6) << progressPercent << "%  \n";
                 cout << "Checked Keys (Total):    " << currentGlobalKeyCount << "         \n";
                 cout << "Speed (approx):          " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s   \n";
                 cout << "Elapsed Time:            " << formatElapsedTime(globalElapsedTime) << "       \n";
                 // Maybe add current rank being checked by thread 0?
                 // cout << "Thread 0 Rank:           " << to_string_128(current_rank.load()) << "        \n";
                 cout.flush();
             }
        }

        // Move to the next combination
        if (!gen.next()) {
            break; // No more combinations in the range assigned to this thread
        }
        current_rank.increment();

    } // End while loop (!stop_event && current_rank < end_rank)

    // Final update of global count from this thread
    globalComparedCount += thread_key_count;
}

// Print command-line usage instructions
void printUsage(const char* programName) {
    cout << "Usage: " << programName << " -p <puzzle_num> [options]\n";
    cout << "Options:\n";
    cout << "  -p, --puzzle NUM    Puzzle number to solve (required, e.g., 20, 68)\n";
    cout << "  -t, --threads NUM   Number of CPU cores to use (default: all available)\n";
    cout << "  -f, --flips NUM     Override default flip count for the puzzle\n";
    cout << "  -k, --key KEY       Specify the base private key manually (decimal string)\n";
    cout << "                      (Overrides the key associated with puzzle -p)\n";
    cout << "  -h, --help          Show this help message\n";
    cout << "\nExample (using puzzle defaults):\n";
    cout << "  " << programName << " -p 68 -t 8\n";
    cout << "\nExample (overriding flips):\n";
    cout << "  " << programName << " -p 68 -t 8 -f 30\n";
    cout << "\nExample (manual base key):\n";
    cout << "  " << programName << " -p 68 -f 34 -k 132656943602386256302\n";
    cout << "\nSupported Puzzle Numbers in PUZZLE_DATA: ";
    bool first = true;
    for(const auto& pair : PUZZLE_DATA) {
        if (!first) cout << ", ";
        cout << pair.first;
        first = false;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    // Setup signal handler for Ctrl+C
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler); // Handle termination signal too

    initConsole(); // Prepare console for special output formatting

    string manual_base_key_input = ""; // Store manually provided key

    // --- Argument Parsing ---
    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"puzzle", required_argument, 0, 'p'},
        {"threads", required_argument, 0, 't'},
        {"flips", required_argument, 0, 'f'},
        {"key", required_argument, 0, 'k'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    // Optstring includes p: (required), t:, f:, k:, h (no arg)
    while ((opt = getopt_long(argc, argv, "p:t:f:k:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                try {
                    PUZZLE_NUM = std::stoi(optarg);
                } catch (const std::exception& e) {
                    cerr << "Error: Invalid puzzle number format: " << optarg << endl;
                    printUsage(argv[0]);
                    return 1;
                }
                // Validate puzzle number exists in our data map later
                break;
            case 't':
                 try {
                    WORKERS = std::stoi(optarg);
                } catch (const std::exception& e) {
                    cerr << "Error: Invalid thread number format: " << optarg << endl;
                    return 1;
                }
                if (WORKERS < 1) {
                    cerr << "Error: Thread count must be at least 1\n";
                    return 1;
                }
                // Cap threads to hardware concurrency if desired, or let user specify more
                // WORKERS = std::min(WORKERS, (int)std::thread::hardware_concurrency());
                // if (WORKERS == 0) WORKERS = 1; // Ensure at least one if detection fails
                break;
            case 'f':
                 try {
                    FLIP_COUNT = std::stoi(optarg);
                } catch (const std::exception& e) {
                    cerr << "Error: Invalid flip count format: " << optarg << endl;
                    return 1;
                }
                if (FLIP_COUNT < 0) {
                    cerr << "Error: Flip count must be non-negative\n";
                    return 1;
                }
                break;
            case 'k':
                manual_base_key_input = optarg;
                 // Basic validation: check if it contains non-digit characters
                 if (manual_base_key_input.find_first_not_of("0123456789") != std::string::npos) {
                     cerr << "Error: Manual key must be a decimal string (only digits 0-9 allowed).\n";
                     return 1;
                 }
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            case '?': // Handle unknown options or missing arguments by getopt
                 // getopt_long usually prints its own error message
                 printUsage(argv[0]);
                 return 1;
            default: // Should not happen
                abort();
        }
    }

     // Check if required puzzle number was provided
     if (PUZZLE_NUM == -1) {
         cerr << "Error: Puzzle number (-p) is required.\n";
         printUsage(argv[0]);
         return 1;
     }

     // Check if the provided puzzle number is in our known data
     auto puzzle_it = PUZZLE_DATA.find(PUZZLE_NUM);
     if (puzzle_it == PUZZLE_DATA.end()) {
         cerr << "Error: Puzzle number " << PUZZLE_NUM << " is not defined in the internal PUZZLE_DATA map.\n";
         printUsage(argv[0]);
         return 1;
     }

    // --- Initialization ---
    tStart = chrono::high_resolution_clock::now();
    Secp256K1 secp;
    secp.Init();

    // Retrieve puzzle data
    auto [DEFAULT_FLIP_COUNT, TARGET_HASH160_HEX, PRIVATE_KEY_DECIMAL_FROM_MAP] = puzzle_it->second;

    // Determine final flip count (use override if provided)
    if (FLIP_COUNT == -1) {
        FLIP_COUNT = DEFAULT_FLIP_COUNT;
    }
    // Validate flip count against puzzle bit length
    if (FLIP_COUNT > PUZZLE_NUM) {
        cerr << "Warning: Flip count (" << FLIP_COUNT
             << ") is greater than puzzle bit length (" << PUZZLE_NUM
             << "). Calculations might be excessive or incorrect.\n";
        // Optionally, cap it:
        // cerr << "Setting flip count to " << PUZZLE_NUM << ".\n";
        // FLIP_COUNT = PUZZLE_NUM;
    }
     if (FLIP_COUNT < 0) {
          cerr << "Internal Error: Flip count became negative after processing defaults/overrides." << endl;
          return 1; // Should have been caught earlier
     }

    // Set and convert target hash
    TARGET_HASH160 = TARGET_HASH160_HEX;
    if (TARGET_HASH160.length() != 40) {
        cerr << "Error: Invalid target HASH160 format in PUZZLE_DATA for puzzle " << PUZZLE_NUM << endl;
        return 1;
    }
    for (size_t i = 0; i < 20; i++) {
        try {
           TARGET_HASH160_RAW[i] = stoul(TARGET_HASH160.substr(i * 2, 2), nullptr, 16);
        } catch (const std::exception& e) {
             cerr << "Error converting target HASH160 hex '" << TARGET_HASH160.substr(i * 2, 2)
                  << "' to byte: " << e.what() << endl;
             return 1;
        }
    }

    // --- Set Base Key (Manual or Default) ---
    string used_base_key_decimal; // Store the key actually used for printing/logging
    if (!manual_base_key_input.empty()) {
        // Use manually provided key
        std::vector<char> manual_key_cstr(manual_base_key_input.c_str(), manual_base_key_input.c_str() + manual_base_key_input.size() + 1); // +1 for null terminator
        BASE_KEY.SetBase10(manual_key_cstr.data()); // Pass mutable buffer
        // Assuming SetBase10 doesn't return status and handles errors internally or crashes
        used_base_key_decimal = manual_base_key_input;
        cout << "Using manual Base Key (Decimal): " << used_base_key_decimal << "\n";
    } else {
        // Use key from puzzle data
        std::vector<char> map_key_cstr(PRIVATE_KEY_DECIMAL_FROM_MAP.c_str(), PRIVATE_KEY_DECIMAL_FROM_MAP.c_str() + PRIVATE_KEY_DECIMAL_FROM_MAP.size() + 1);
        BASE_KEY.SetBase10(map_key_cstr.data()); // Pass mutable buffer
        // Assuming SetBase10 is successful for map data
        used_base_key_decimal = PRIVATE_KEY_DECIMAL_FROM_MAP;
        // Optional feedback: cout << "Using Base Key from Puzzle Data (Decimal): " << used_base_key_decimal << "\n";
    }

    // Optional: Check if base key seems too large for the puzzle context (informational)
    if (BASE_KEY.GetBitLength() > PUZZLE_NUM) {
        cerr << "Warning: Effective Base Key (" << BASE_KEY.GetBitLength()
             << " bits) exceeds puzzle bit length (" << PUZZLE_NUM << " bits).\n";
    }

    // Calculate total combinations
    total_combinations = CombinationGenerator::combinations_count(PUZZLE_NUM, FLIP_COUNT);
    if (total_combinations == static_cast<__uint128_t>(-1)) { // Check for overflow indicator
        cerr << "Error: Calculation of total combinations resulted in overflow. Cannot proceed.\n";
        return 1;
    }
     if (total_combinations == 0 && FLIP_COUNT > 0) {
          cerr << "Warning: Calculated 0 combinations for n=" << PUZZLE_NUM << ", k=" << FLIP_COUNT << ". Check parameters.\n";
          // Allow proceeding if k=0, otherwise it might be an error state.
          if (FLIP_COUNT > 0) return 1;
     }

    // Format base key (hex) for display (pad to puzzle bit length)
    string baseKeyHex = BASE_KEY.GetBase16();
    int requiredHexChars = (PUZZLE_NUM + 3) / 4; // Calculate needed hex chars, round up
    baseKeyHex = string(max(0, requiredHexChars - (int)baseKeyHex.length()), '0') + baseKeyHex.substr(max(0, (int)baseKeyHex.length() - requiredHexChars));


    // --- Print Header ---
    clearTerminal();
    cout << "=======================================\n";
    cout << "== Mutagen Puzzle Solver             ==\n";
    cout << "=======================================\n";
    cout << "Puzzle Number:    " << PUZZLE_NUM << " (" << PUZZLE_NUM << "-bit)\n";
    cout << "Target HASH160:   " << TARGET_HASH160 << "\n";
    cout << "Base Key (Hex):   0x" << baseKeyHex << "\n";
    if (!manual_base_key_input.empty()){
         cout << "Base Key (Dec):   " << used_base_key_decimal << " (Manual Input)\n";
    } else {
         cout << "Base Key (Dec):   " << used_base_key_decimal << " (From Puzzle Data)\n";
    }
    cout << "Flip Count:       " << FLIP_COUNT;
    if (FLIP_COUNT != DEFAULT_FLIP_COUNT && manual_base_key_input.empty()) {
        cout << " (Override, default was " << DEFAULT_FLIP_COUNT << ")";
    } else if (FLIP_COUNT != DEFAULT_FLIP_COUNT && !manual_base_key_input.empty()) {
        cout << " (Override)"; // Indicate override even with manual key if different from default
    }
    cout << "\n";
    cout << "Total Combinations:" << (total_combinations == static_cast<__uint128_t>(-1) ? "OVERFLOW" : to_string_128(total_combinations)) << "\n";
    cout << "Using Threads:    " << WORKERS << "\n";
    cout << "---------------------------------------\n"; // Separator before progress

    // --- Thread Creation and Work Distribution ---
    vector<thread> threads;
    AVXCounter total_combinations_avx;
    total_combinations_avx.store(total_combinations);

    if (total_combinations == 0) {
        // Handle k=0 case: Only need to check offsets from the base key
        // The current worker loop structure handles this implicitly,
        // as CombinationGenerator(n, 0) should yield one empty flip vector.
        // We still need to distribute the "single combination check" across threads,
        // though it's less efficient. For simplicity, run it on one thread or adapt worker.
        cout << "\nFlip count is 0. Checking offsets from the base key...\n";
        // For now, let the standard distribution handle this (it will run one 'combination').
        // A dedicated k=0 path might be slightly faster but adds complexity.
        // WORKERS = 1; // Option: Force single thread for k=0?
        if (WORKERS > 1) {
             cout << "Warning: Using multiple threads for k=0 might be inefficient.\n";
        }
         // The existing distribution logic should work, assigning rank 0 to thread 0.
         total_combinations = 1; // Treat k=0 as 1 combination for distribution
         total_combinations_avx.store(1);
    }


    if (total_combinations > 0 && total_combinations != static_cast<__uint128_t>(-1))
    {
        AVXCounter comb_per_thread = AVXCounter::div(total_combinations_avx, WORKERS);
        uint64_t remainder = AVXCounter::mod(total_combinations_avx, WORKERS);
        AVXCounter current_start_rank; // Starts at 0

        for (int i = 0; i < WORKERS; i++) {
            AVXCounter assigned_count = comb_per_thread;
            if (i < remainder) {
                assigned_count.increment(); // Add one from the remainder
            }

            AVXCounter start_rank = current_start_rank;
            AVXCounter end_rank = start_rank;
            end_rank.add(assigned_count.load());

            // Ensure end_rank doesn't exceed total combinations if count is > 0
             if (assigned_count.load() > 0 && end_rank.load() > total_combinations) {
                 end_rank.store(total_combinations);
             }

            // Launch thread only if there's work in the range [start_rank, end_rank)
             if (start_rank < end_rank) {
                 threads.emplace_back(worker, &secp, PUZZLE_NUM, FLIP_COUNT, i, start_rank, end_rank);
             }

            current_start_rank = end_rank; // Set start for the next thread
        }
    } else if (total_combinations != static_cast<__uint128_t>(-1)) { // Only print if not overflow
         cout << "\nNo combinations to check (total_combinations is 0 and flip_count > 0?).\n";
         return 0; // Exit cleanly
    }


    // --- Wait for Threads and Report Result ---
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto tEnd = chrono::high_resolution_clock::now();
    globalElapsedTime = chrono::duration<double>(tEnd - tStart).count();
    uint64_t finalKeyCount = globalComparedCount.load();
    if (globalElapsedTime > 0.01) {
       mkeysPerSec = (double)finalKeyCount / globalElapsedTime / 1e6;
    } else {
        mkeysPerSec = 0.0; // Avoid division by zero or inf values
    }

    // Final progress update to show 100% if completed without finding
    // Ensure cursor is below the potentially multi-line progress area
    moveCursorTo(0, 15); // Move below the progress lines

    if (!results.empty()) {
        // --- Solution Found ---
        auto [hex_key, comb_index, flips] = results.front();

        // Format result key for display (compact hex)
        string compactHex = hex_key;
        size_t firstNonZero = compactHex.find_first_not_of('0');
        if (string::npos != firstNonZero) {
             compactHex = compactHex.substr(firstNonZero);
        } else {
             compactHex = "0"; // Handle case where key is literally 0
        }
         compactHex = "0x" + compactHex;


        cout << "\n=======================================\n";
        cout << "=========== SOLUTION FOUND ============\n";
        cout << "=======================================\n";
        cout << "Private key (Hex): " << compactHex << "\n"; // Compact hex
        cout << "Private key (Dec): ";
        // --- FIX: Convert hex_key string to Int using mutable copy ---
        Int resultKey;
        std::vector<char> hex_key_cstr(hex_key.c_str(), hex_key.c_str() + hex_key.size() + 1);
        resultKey.SetBase16(hex_key_cstr.data()); // Pass mutable buffer
        cout << resultKey.GetBase10() << "\n";
        // --- End Fix ---
        cout << "Found at comb idx: " << to_string_128(comb_index) << " (approx)\n"; // Rank where solution was found
        cout << "Bit flips used:    " << flips << "\n";
        cout << "Total Keys Checked:" << finalKeyCount << "\n";
        cout << "Time:              " << fixed << setprecision(2) << globalElapsedTime << " seconds ("
             << formatElapsedTime(globalElapsedTime) << ")\n";
        cout << "Average Speed:     " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s\n";

        // Save solution (full, padded hex key)
        ofstream out("puzzle_" + to_string(PUZZLE_NUM) + "_solution.txt");
        if (out) {
            out << hex_key; // Save the full 64-char hex key
            out.close();
            cout << "Solution saved to puzzle_" << PUZZLE_NUM << "_solution.txt\n";
        } else {
            cerr << "Failed to save solution to file!\n";
        }
    } else if (stop_event.load()) {
        // --- Search Interrupted ---
         cout << "\n\nSearch stopped prematurely (Ctrl+C or error).\n";
         cout << "Checked " << finalKeyCount << " keys in " << fixed << setprecision(2) << globalElapsedTime << " seconds.\n";
         cout << "Average Speed: " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s\n";
    }
     else {
        // --- No Solution Found ---
        cout << "\n\nNo solution found after checking all combinations.\n";
         if (total_combinations != static_cast<__uint128_t>(-1)) {
             cout << "Checked " << to_string_128(total_combinations) << " combinations (" << finalKeyCount << " keys).\n";
         } else {
              cout << "Checked " << finalKeyCount << " keys (combination count overflowed).\n";
         }
        cout << "Time: " << fixed << setprecision(2) << globalElapsedTime << " seconds ("
             << formatElapsedTime(globalElapsedTime) << ")\n";
        cout << "Average Speed: " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s\n";
    }

    return results.empty() ? 1 : 0; // Return 0 if found, 1 if not found or interrupted
}
