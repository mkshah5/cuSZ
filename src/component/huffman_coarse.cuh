/**
 * @file huffman_coarse.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_COMPONENT_HUFFMAN_COARSE_CUH
#define CUSZ_COMPONENT_HUFFMAN_COARSE_CUH

#include <cuda.h>
#include <clocale>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <type_traits>
using std::cout;

#include "../../include/reducer.hh"
#include "../common/capsule.hh"
#include "../common/definition.hh"
#include "../common/type_traits.hh"
#include "../kernel/codec_huffman.cuh"
#include "../kernel/hist.cuh"
#include "../utils.hh"
#include "huffman_coarse.cuh"
#include "huffman_parbook.cuh"

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define EXPORT_NBYTE(FIELD) nbyte[HEADER::FIELD] = rte.nbyte[RTE::FIELD];

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                            \
    {                                                                             \
        constexpr auto D2D = cudaMemcpyDeviceToDevice;                            \
        auto           dst = d_compressed + header.entry[HEADER::FIELD];          \
        auto           src = reinterpret_cast<BYTE*>(d_##VAR);                    \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], D2D, stream)); \
    }

#define DEFINE_HC_ARRAY(VAR, TYPE) \
    TYPE* d_##VAR{nullptr};        \
    TYPE* h_##VAR{nullptr};

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[HEADER::SYM])

#define HC_ALLOCHOST(VAR, SYM)                     \
    cudaMallocHost(&h_##VAR, rte.nbyte[RTE::SYM]); \
    memset(h_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_ALLOCDEV(VAR, SYM)                  \
    cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM]); \
    cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_FREEHOST(VAR)       \
    if (h_##VAR) {             \
        cudaFreeHost(h_##VAR); \
        h_##VAR = nullptr;     \
    }

#define HC_FREEDEV(VAR)    \
    if (d_##VAR) {         \
        cudaFree(d_##VAR); \
        d_##VAR = nullptr; \
    }

/******************************************************************************
                                class definition
 ******************************************************************************/

namespace cusz {

template <typename Huff, typename Meta>
__global__ void
huffman_coarse_concatenate(Huff* gapped, Meta* par_entry, Meta* par_ncell, int const cfg_sublen, Huff* non_gapped)
{
    auto n   = par_ncell[blockIdx.x];
    auto src = gapped + cfg_sublen * blockIdx.x;
    auto dst = non_gapped + par_entry[blockIdx.x];

    for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
        dst[i] = src[i];
    }
}

}  // namespace cusz

namespace cusz {

template <typename T, typename H, typename M = uint32_t>
class HuffmanCoarse : public cusz::VariableRate {
   public:
    using Origin    = T;
    using Encoded   = H;
    using MetadataT = M;
    using FreqT     = cusz::FREQ;
    using BYTE      = uint8_t;

   private:
    DEFINE_HC_ARRAY(tmp, H);
    DEFINE_HC_ARRAY(compressed, BYTE);  // alias in address
    DEFINE_HC_ARRAY(freq, FreqT);
    DEFINE_HC_ARRAY(book, H);
    DEFINE_HC_ARRAY(revbook, BYTE);
    DEFINE_HC_ARRAY(par_nbit, M);
    DEFINE_HC_ARRAY(par_ncell, M);
    DEFINE_HC_ARRAY(par_entry, M);
    DEFINE_HC_ARRAY(bitstream, H);

   public:
    /**
     * @brief on host; separate from (binary) data fields
     * otherwise, aligning to 128B can be unwanted
     *
     */
    struct header_t {
        static const int HEADER    = 0;
        static const int REVBOOK   = 1;
        static const int PAR_NBIT  = 2;
        static const int PAR_ENTRY = 3;
        static const int BITSTREAM = 4;
        static const int END       = 5;

        int       header_nbyte : 16;
        int       booklen : 16;
        int       sublen;
        int       pardeg;
        size_t    uncompressed_len;
        size_t    total_nbit;
        size_t    total_ncell;  // TODO change to uint32_t
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };
    using HEADER = header_t;

    struct runtime_encode_helper {
        static const int TMP       = 0;
        static const int FREQ      = 1;
        static const int BOOK      = 2;
        static const int REVBOOK   = 3;
        static const int PAR_NBIT  = 4;
        static const int PAR_NCELL = 5;
        static const int PAR_ENTRY = 6;
        static const int BITSTREAM = 7;
        static const int END       = 8;

        uint32_t nbyte[END];
    };
    using RTE = runtime_encode_helper;
    RTE rte;

    void dbg_println(const std::string SYM_name, void* VAR, int SYM)
    {
        CUdeviceptr pbase0{0};
        size_t      psize0{0};

        cuMemGetAddressRange(&pbase0, &psize0, (CUdeviceptr)VAR);
        printf(
            "%s:\n"
            "\t(supposed) pointer : %p\n"
            "\t(supposed) bytes   : %'9lu\n"
            "\t(queried)  pbase0  : %p\n"
            "\t(queried)  psize0  : %'9lu\n",
            SYM_name.c_str(), (void*)VAR, (size_t)rte.nbyte[SYM], (void*)&pbase0, psize0);
        pbase0 = 0, psize0 = 0;
    }

    /**
     * @brief Allocate workspace according to the input size & configurations.
     *
     * @param in_uncompressed_len uncompressed length
     * @param cfg_booklen codebook length
     * @param cfg_pardeg degree of parallelism
     * @param dbg_print print for debugging
     */
    void init(size_t const in_uncompressed_len, int cfg_booklen, int cfg_pardeg, bool dbg_print = false)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / 2 * sizeof(H); };

        auto debug = [&]() {
            setlocale(LC_NUMERIC, "");

            printf("\nHuffmanCoarse::init() debugging:\n");
            printf("CUdeviceptr nbyte: %d\n", (int)sizeof(CUdeviceptr));

            dbg_println("TMP", d_tmp, RTE::TMP);
            dbg_println("FREQ", d_freq, RTE::FREQ);
            dbg_println("BOOK", d_book, RTE::BOOK);
            dbg_println("REVBOOK", d_revbook, RTE::REVBOOK);
            dbg_println("PAR_NBIT", d_par_nbit, RTE::PAR_NBIT);
            dbg_println("PAR_NCELL", d_par_ncell, RTE::PAR_NCELL);
            dbg_println("BITSTREAM", d_bitstream, RTE::BITSTREAM);
            printf("\n");
        };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
        // memset(rte.entry, 0, sizeof(uint32_t) * (RTE::END + 1));

        rte.nbyte[RTE::TMP]       = sizeof(H) * in_uncompressed_len;
        rte.nbyte[RTE::FREQ]      = sizeof(FreqT) * cfg_booklen;
        rte.nbyte[RTE::BOOK]      = sizeof(H) * cfg_booklen;
        rte.nbyte[RTE::REVBOOK]   = get_revbook_nbyte(cfg_booklen);
        rte.nbyte[RTE::PAR_NBIT]  = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::PAR_NCELL] = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::PAR_ENTRY] = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::BITSTREAM] = max_compressed_bytes();

        HC_ALLOCDEV(tmp, TMP);
        HC_ALLOCDEV(freq, FREQ);
        HC_ALLOCDEV(book, BOOK);
        HC_ALLOCDEV(revbook, REVBOOK);
        HC_ALLOCDEV(par_nbit, PAR_NBIT);
        HC_ALLOCDEV(par_ncell, PAR_NCELL);
        HC_ALLOCDEV(par_entry, PAR_ENTRY);
        HC_ALLOCDEV(bitstream, BITSTREAM);

        // standalone definition for output
        d_compressed = reinterpret_cast<BYTE*>(d_tmp);

        HC_ALLOCHOST(freq, FREQ);
        HC_ALLOCHOST(book, BOOK);
        HC_ALLOCHOST(revbook, REVBOOK);
        HC_ALLOCHOST(par_nbit, PAR_NBIT);
        HC_ALLOCHOST(par_ncell, PAR_NCELL);
        HC_ALLOCHOST(par_entry, PAR_ENTRY);

        if (dbg_print) debug();
    }

   private:
    using BOOK = H;
    using SYM  = T;

    static const int CELL_BITWIDTH = sizeof(H) * 8;

    float milliseconds{0.0};
    float time_hist{0.0}, time_book{0.0}, time_lossless{0.0};

   public:
    //
    float get_time_elapsed() const { return milliseconds; }
    float get_time_hist() const { return time_hist; }
    float get_time_book() const { return time_book; }
    float get_time_lossless() const { return time_lossless; }

    H*          expose_book() const { return d_book; };
    BYTE*       expose_revbook() const { return d_revbook; };
    cusz::FREQ* expose_freq() const { return d_freq; };

    // TODO this kind of space will be overlapping with quant-codes
    size_t get_workspace_nbyte(size_t len) const { return sizeof(H) * len; }
    size_t get_max_output_nbyte(size_t len) const { return sizeof(H) * len / 2; }

    static uint32_t get_revbook_nbyte(int dict_size)
    {
        return sizeof(BOOK) * (2 * CELL_BITWIDTH) + sizeof(SYM) * dict_size;
    }

    constexpr bool can_overlap_input_and_firstphase_encode() { return sizeof(T) == sizeof(H); }

   public:
    // 21-12-17 toward static method
    HuffmanCoarse() = default;

    ~HuffmanCoarse()
    {
        HC_FREEDEV(tmp);
        HC_FREEDEV(freq);
        HC_FREEDEV(book);
        HC_FREEDEV(revbook);
        HC_FREEDEV(par_nbit);
        HC_FREEDEV(par_ncell);
        HC_FREEDEV(par_entry);
        HC_FREEDEV(bitstream);

        HC_FREEHOST(freq);
        HC_FREEHOST(book);
        HC_FREEHOST(revbook);
        HC_FREEHOST(par_nbit);
        HC_FREEHOST(par_ncell);
        HC_FREEHOST(par_entry);
    }

   public:

   void generate_precreated_files(int booklen){
        /** Cauchy entropies  **/

        // float entropy_vals[100] = {0.32,0.345,0.372,0.401,0.432,0.465,0.5,0.537,0.577,0.62,0.664,0.712,0.762,0.815,0.871,0.929,0.991,1.055,1.122,1.191,1.264,1.339,1.416,1.495,1.577,1.66,1.745,1.831,1.918,2.005,2.094,2.182,2.271,2.359,2.446,2.533,2.619,2.703,2.786,2.868,2.949,3.028,3.106,3.182,3.257,3.33,3.403,3.474,3.545,3.615,3.683,3.752,3.819,3.887,3.954,4.02,4.087,4.153,4.219,4.285,4.35,4.416,4.482,4.547,4.612,4.678,4.743,4.808,4.873,4.938,5.003,5.068,5.132,5.197,5.262,5.326,5.39,5.455,5.519,5.583,5.647,5.71,5.774,5.838,5.901,5.964,6.027,6.09,6.153,6.216,6.278,6.34,6.402,6.464,6.526,6.588,6.649,6.71,6.771,6.832};
        
        /** Gaussian entropies **/
        // float entropies[100] = {1.016,1.075,1.131,1.184,1.236,1.286,1.335,1.382,1.428,1.474,1.518,1.563,1.607,1.651,1.695,1.738,1.782,1.826,1.869,1.913,1.957,2.0,2.044,2.088,2.131,2.175,2.219,2.262,2.306,2.35,2.393,2.437,2.481,2.524,2.568,2.611,2.655,2.699,2.742,2.786,2.83,2.873,2.917,2.961,3.004,3.048,3.092,3.135,3.179,3.223,3.266,3.31,3.354,3.397,3.441,3.484,3.528,3.572,3.615,3.659,3.703,3.746,3.79,3.834,3.877,3.921,3.965,4.008,4.052,4.096,4.139,4.183,4.227,4.27,4.314,4.357,4.401,4.445,4.488,4.532,4.576,4.619,4.663,4.707,4.75,4.794,4.838,4.881,4.925,4.969,5.012,5.056,5.099,5.143,5.187,5.23,5.274,5.318,5.361,5.405};
        
        
        /** Laplace entropies **/
        float entropy_vals[100] = {0.002,0.003,0.005,0.006,0.009,0.012,0.016,0.021,0.027,0.036,0.046,0.058,0.072,0.089,0.109,0.132,0.158,0.188,0.221,0.258,0.299,0.343,0.391,0.442,0.497,0.555,0.616,0.68,0.746,0.815,0.885,0.958,1.032,1.107,1.183,1.261,1.338,1.416,1.495,1.573,1.652,1.73,1.808,1.886,1.963,2.04,2.117,2.193,2.269,2.344,2.419,2.493,2.567,2.64,2.713,2.786,2.858,2.93,3.001,3.072,3.143,3.214,3.284,3.354,3.423,3.493,3.562,3.631,3.7,3.769,3.838,3.907,3.975,4.043,4.111,4.18,4.248,4.315,4.383,4.451,4.519,4.587,4.654,4.722,4.789,4.857,4.924,4.992,5.059,5.126,5.194,5.261,5.328,5.396,5.463,5.53,5.597,5.665,5.732,5.799};
        
        int num_buckets = 1024;
        uint32_t *large_hist_arr = (uint32_t *)malloc(sizeof(uint32_t)*num_buckets*NUM_TREES);
        float *large_logq_arr = (float *)malloc(sizeof(float)*num_buckets*NUM_TREES);

        for (size_t i = 0; i < NUM_TREES; i++) // Read in every histogram and store to large hist array
        {
            char entropy_file[100];
            sprintf(entropy_file, "laplace_hists/hist_entropy_%0.3f.data", entropy_vals[(100/NUM_TREES)*i]);

            FILE *q_file = fopen(entropy_file,"rb");
            fread((void *)&large_hist_arr[i*num_buckets], sizeof(uint32_t), num_buckets, q_file);
            fclose(q_file);


        }

        for (size_t i = 0; i < NUM_TREES; i++) // Calculate all the log2(q) values beforehand, then only need a multiply-accumulate
        {
            int q_total = 0;
            for (size_t j = 0; j < num_buckets; j++)
            {
                q_total+=large_hist_arr[i*num_buckets+j];
            }

            for (size_t j = 0; j < num_buckets; j++)
            {
                uint32_t hist_value = large_hist_arr[i*num_buckets+j];
                large_logq_arr[i*num_buckets+j] = log2((double)hist_value/(double)q_total);
            }
        }

        FILE *logq_file = fopen("laplace_logq.bin", "wb");
        fwrite(large_logq_arr, sizeof(float), num_buckets*NUM_TREES, logq_file);
        fclose(logq_file);


   }
    /**
     * @brief Inspect the input data; generate histogram, codebook (for encoding), reversed codebook (for decoding).
     *
     * @param tmp_freq (device array) If called by other class methods, use class-private d_freq; otherwise, use
     * external array.
     * @param tmp_book (device array) If called by other class methods, use class-private d_book.
     * @param in_uncompressed (device array) input data
     * @param in_uncompressed_len (host variable) input data length
     * @param cfg_booklen (host variable) configuration, book size
     * @param out_revbook (device array) If called by other class methods, use class-private d_revbook.
     * @param stream CUDA stream
     */
    void inspect(
        cusz::FREQ*  tmp_freq,
        H*           tmp_book,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        BYTE*        out_revbook,
        cudaStream_t stream = nullptr,
        float        entropy_use = 1.016)
    {
        kernel_wrapper::get_frequency<T>(
            in_uncompressed, in_uncompressed_len, tmp_freq, cfg_booklen, time_hist, stream, entropy_use);

        // This is end-to-end time for parbook.
        cuda_timer_t t;
        t.timer_start(stream);
        kernel_wrapper::par_get_codebook<T, H>(tmp_freq, cfg_booklen, tmp_book, out_revbook, stream);
        t.timer_end(stream);
        cudaStreamSynchronize(stream);

        time_book = t.get_time_elapsed();

        generate_precreated_files(cfg_booklen);    

    }


   private:
    /**
     * @brief Collect fragmented field with repurposing TMP space.
     *
     * @param header (host variable)
     * @param stream CUDA stream
     */
    void subfile_collect(
        HEADER&      header,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        int const    cfg_pardeg,
        cudaStream_t stream = nullptr)
    {
        auto BARRIER = [&]() {
            if (stream)
                CHECK_CUDA(cudaStreamSynchronize(stream));
            else
                CHECK_CUDA(cudaDeviceSynchronize());
        };

        header.header_nbyte     = sizeof(struct header_t);
        header.booklen          = cfg_booklen;
        header.sublen           = cfg_sublen;
        header.pardeg           = cfg_pardeg;
        header.uncompressed_len = in_uncompressed_len;

        MetadataT nbyte[HEADER::END];
        nbyte[HEADER::HEADER] = 128;

        EXPORT_NBYTE(REVBOOK)
        EXPORT_NBYTE(PAR_NBIT)
        EXPORT_NBYTE(PAR_ENTRY)
        EXPORT_NBYTE(BITSTREAM)

        header.entry[0] = 0;
        // *.END + 1: need to know the ending position
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        auto debug_header_entry = [&]() {
            for (auto i = 0; i < HEADER::END + 1; i++) printf("%d, header entry: %d\n", i, header.entry[i]);
        };
        // debug_header_entry();

        CHECK_CUDA(cudaMemcpyAsync(d_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

        /* debug */ BARRIER();

        DEVICE2DEVICE_COPY(revbook, REVBOOK)
        DEVICE2DEVICE_COPY(par_nbit, PAR_NBIT)
        DEVICE2DEVICE_COPY(par_entry, PAR_ENTRY)
        DEVICE2DEVICE_COPY(bitstream, BITSTREAM)
    }

   public:
    /**
     * @brief
     *
     */
    void clear_buffer()
    {
        cudaMemset(d_tmp, 0x0, rte.nbyte[RTE::TMP]);
        cudaMemset(d_freq, 0x0, rte.nbyte[RTE::FREQ]);
        cudaMemset(d_book, 0x0, rte.nbyte[RTE::BOOK]);
        cudaMemset(d_revbook, 0x0, rte.nbyte[RTE::REVBOOK]);
        cudaMemset(d_par_nbit, 0x0, rte.nbyte[RTE::PAR_NBIT]);
        cudaMemset(d_par_ncell, 0x0, rte.nbyte[RTE::PAR_NCELL]);
        cudaMemset(d_par_entry, 0x0, rte.nbyte[RTE::PAR_ENTRY]);
        cudaMemset(d_bitstream, 0x0, rte.nbyte[RTE::BITSTREAM]);
    }

    /**
     * @brief Public encode interface.
     *
     * @param in_uncompressed (device array)
     * @param in_uncompressed_len (host variable)
     * @param cfg_booklen (host variable)
     * @param cfg_sublen (host variable)
     * @param out_compressed (device array) reference
     * @param out_compressed_len (host variable) reference output
     * @param stream CUDA stream
     */
    void encode(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        int const    cfg_pardeg,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream     = nullptr,
        bool         do_inspect = true,
        float        entropy_use = 1.016)
    {
        cuda_timer_t t;
        time_lossless = 0;

        auto BARRIER = [&]() {
            if (stream)
                CHECK_CUDA(cudaStreamSynchronize(stream));
            else
                CHECK_CUDA(cudaDeviceSynchronize());
        };

        struct header_t header;

        auto encode_phase1 = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
            auto grid_dim  = ConfigHelper::get_npart(in_uncompressed_len, block_dim);

            int numSMs;
            cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_fixedlen_gridstride<T, H>
                <<<8 * numSMs, 256, sizeof(H) * cfg_booklen, stream>>>  //
                (in_uncompressed, in_uncompressed_len, d_book, cfg_booklen, d_tmp);

            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        auto encode_phase2 = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
            auto grid_dim  = ConfigHelper::get_npart(cfg_pardeg, block_dim);

            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_deflate<H><<<grid_dim, block_dim, 0, stream>>>  //
                (d_tmp, in_uncompressed_len, d_par_nbit, d_par_ncell, cfg_sublen, cfg_pardeg);

            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        auto encode_phase3 = [&]() {
            CHECK_CUDA(cudaMemcpyAsync(h_par_nbit, d_par_nbit, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(
                cudaMemcpyAsync(h_par_ncell, d_par_ncell, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
            BARRIER();

            memcpy(h_par_entry + 1, h_par_ncell, (cfg_pardeg - 1) * sizeof(M));
            for (auto i = 1; i < cfg_pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

            header.total_nbit  = std::accumulate(h_par_nbit, h_par_nbit + cfg_pardeg, (size_t)0);
            header.total_ncell = std::accumulate(h_par_ncell, h_par_ncell + cfg_pardeg, (size_t)0);

            CHECK_CUDA(
                cudaMemcpyAsync(d_par_entry, h_par_entry, cfg_pardeg * sizeof(M), cudaMemcpyHostToDevice, stream));
            BARRIER();

            // update with the precise BITSTREAM nbyte
            rte.nbyte[RTE::BITSTREAM] = sizeof(H) * header.total_ncell;
        };

        auto encode_phase4 = [&]() {
            t.timer_start(stream);
            {
                cusz::huffman_coarse_concatenate<H, M><<<cfg_pardeg, 128, 0, stream>>>  //
                    (d_tmp, d_par_entry, d_par_ncell, cfg_sublen, d_bitstream);
            }
            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        // -----------------------------------------------------------------------------

        if (do_inspect) inspect(d_freq, d_book, in_uncompressed, in_uncompressed_len, cfg_booklen, d_revbook, stream, entropy_use);

        encode_phase1();
        encode_phase2();
        encode_phase3();
        encode_phase4();

        subfile_collect(header, in_uncompressed_len, cfg_booklen, cfg_sublen, cfg_pardeg, stream);

        out_compressed     = d_compressed;
        out_compressed_len = header.subfile_size();
    }

    /**
     * @brief Public decode interface.
     *
     * @param in_compressed (device array) input
     * @param out_decompressed (device array output) output
     * @param stream CUDA stream
     * @param header_on_device If true, copy header from device binary to host.
     */
    void decode(
        BYTE*        in_compressed,  //
        T*           out_decompressed,
        cudaStream_t stream           = nullptr,
        bool         header_on_device = true)
    {
        header_t header;
        if (header_on_device)
            CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));

        auto d_revbook   = ACCESSOR(REVBOOK, BYTE);
        auto d_par_nbit  = ACCESSOR(PAR_NBIT, M);
        auto d_par_entry = ACCESSOR(PAR_ENTRY, M);
        auto d_bitstream = ACCESSOR(BITSTREAM, H);

        auto const revbook_nbyte = get_revbook_nbyte(header.booklen);
        auto const pardeg        = header.pardeg;
        auto const block_dim     = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
        auto const grid_dim      = ConfigHelper::get_npart(pardeg, block_dim);

        cuda_timer_t t;
        t.timer_start(stream);
        cusz::coarse_par::detail::kernel::huffman_decode_new<T, H, M><<<grid_dim, block_dim, revbook_nbyte, stream>>>(
            d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, header.sublen, pardeg, out_decompressed);
        t.timer_end(stream);
        cudaStreamSynchronize(stream);

        time_lossless = t.get_time_elapsed();
    }

    // end of class definition
};

}  // namespace cusz

#undef HC_ALLOCDEV
#undef HC_ALLOCHOST
#undef HC_FREEDEV
#undef HC_FREEHOST
#undef EXPORT_NBYTE
#undef ACCESSOR
#undef DEVICE2DEVICE_COPY

#endif
