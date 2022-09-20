/**
 * @file compressor.cuh
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4) 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DEFAULT_PATH_CUH
#define CUSZ_DEFAULT_PATH_CUH

#include "base_compressor.cuh"
#include "binding.hh"
#include "componments.hh"
#include "header.hh"

#define DEFINE_DEV(VAR, TYPE) TYPE* d_##VAR{nullptr};
#define DEFINE_HOST(VAR, TYPE) TYPE* h_##VAR{nullptr};
#define FREEDEV(VAR) CHECK_CUDA(cudaFree(d_##VAR));
#define FREEHOST(VAR) CHECK_CUDA(cudaFreeHost(h_##VAR));

#define D2D_CPY(VAR, FIELD)                                                                            \
    {                                                                                                  \
        auto dst = d_reserved_compressed + header.entry[HEADER::FIELD];                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header->entry[HEADER::SYM])

namespace cusz {

constexpr auto kHOST        = cusz::LOC::HOST;
constexpr auto kDEVICE      = cusz::LOC::DEVICE;
constexpr auto kHOST_DEVICE = cusz::LOC::HOST_DEVICE;

template <class BINDING>
class Compressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor     = typename BINDING::PREDICTOR;
    using SpCodec       = typename BINDING::SPCODEC;
    using Codec         = typename BINDING::CODEC;
    using FallbackCodec = typename BINDING::FALLBACK_CODEC;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;
    using H_FB = typename FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    bool use_fallback_codec{false};
    bool fallback_codec_allocated{false};

    using HEADER = cuszHEADER;
    HEADER header;

    BYTE* d_reserved_compressed{nullptr};

    TimeRecord timerecord;

   private:
    Predictor*     predictor;
    SpCodec*       spcodec;
    Codec*         codec;
    FallbackCodec* fb_codec;

   private:
    dim3     data_len3;
    uint32_t get_len_data() { return data_len3.x * data_len3.y * data_len3.z; }

   public:
    // void export_header(HEADER*& ext_header) { ext_header = &header; }
    void export_header(HEADER& ext_header) { ext_header = header; }

    Compressor()
    {
        predictor = new Predictor;
        spcodec   = new SpCodec;
        codec     = new Codec;
        fb_codec  = new FallbackCodec;
    }

    void destroy()
    {
        if (spcodec) delete spcodec;
        if (codec) delete codec;
        if (fb_codec) delete codec;
        if (predictor) delete predictor;
    }

    ~Compressor() { destroy(); }

    /**
     * @brief Export internal Time Record list by deep copy.
     *
     * @param ext_timerecord nullable; pointer to external TimeRecord.
     */
    void export_timerecord(TimeRecord* ext_timerecord)
    {
        if (ext_timerecord) *ext_timerecord = timerecord;
    }

    template <class CONFIG>
    void init(CONFIG* config, bool dbg_print = false)
    {
        const auto cfg_radius      = (*config).radius;
        const auto cfg_pardeg      = (*config).vle_pardeg;
        const auto density_factor  = (*config).nz_density_factor;
        const auto codec_config    = (*config).codecs_in_use;
        const auto cfg_max_booklen = cfg_radius * 2;
        const auto x               = (*config).x;
        const auto y               = (*config).y;
        const auto z               = (*config).z;

        size_t spcodec_in_len, codec_in_len;

        auto allocate_codec = [&]() {
            if (codec_config == 0b00) throw std::runtime_error("Argument codec_config must have set bit(s).");
            if (codec_config bitand 0b01) {
                LOGGING(LOG_INFO, "allocated 4-byte codec");
                (*codec).init(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
            }
            if (codec_config bitand 0b10) {
                LOGGING(LOG_INFO, "allocated 8-byte (fallback) codec");
                (*fb_codec).init(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
                fallback_codec_allocated = true;
            }
        };

        (*predictor).init(x, y, z, dbg_print);

        spcodec_in_len = (*predictor).get_alloclen_data();
        codec_in_len   = (*predictor).get_alloclen_quant();

        (*spcodec).init(spcodec_in_len, density_factor, dbg_print);

        allocate_codec();

        CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_alloclen_data() * sizeof(T) / 2));
    }

    void collect_compress_timerecord()
    {
#define COLLECT_TIME(NAME, TIME) timerecord.push_back({const_cast<const char*>(NAME), TIME});

        if (not timerecord.empty()) timerecord.clear();

        COLLECT_TIME("predict", (*predictor).get_time_elapsed());

        if (not use_fallback_codec) {
            COLLECT_TIME("histogram", (*codec).get_time_hist());
            COLLECT_TIME("book", (*codec).get_time_book());
            COLLECT_TIME("huff-enc", (*codec).get_time_lossless());
        }
        else {
            COLLECT_TIME("histogram", (*fb_codec).get_time_hist());
            COLLECT_TIME("book", (*fb_codec).get_time_book());
            COLLECT_TIME("huff-enc", (*fb_codec).get_time_lossless());
        }

        COLLECT_TIME("outlier", (*spcodec).get_time_elapsed());
    }

    void collect_decompress_timerecord()
    {
        if (not timerecord.empty()) timerecord.clear();

        COLLECT_TIME("outlier", (*spcodec).get_time_elapsed());

        if (not use_fallback_codec) {  //
            COLLECT_TIME("huff-dec", (*codec).get_time_lossless());
        }
        else {  //
            COLLECT_TIME("huff-dec", (*fb_codec).get_time_lossless());
        }

        COLLECT_TIME("predict", (*predictor).get_time_elapsed());
    }

    template <class CONFIG>
    void compress(
        CONFIG*      config,
        T*           uncompressed,
        BYTE*&       compressed,
        size_t&      compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        auto const eb                = (*config).eb;
        auto const radius            = (*config).radius;
        auto const pardeg            = (*config).vle_pardeg;
        auto const codecs_in_use     = (*config).codecs_in_use;
        auto const nz_density_factor = (*config).nz_density_factor;

        data_len3 = dim3((*config).x, (*config).y, (*config).z);

        compress_detail(
            uncompressed, eb, radius, pardeg, codecs_in_use, nz_density_factor, compressed, compressed_len,
            (*config).codec_force_fallback(), stream, dbg_print);
    }

    void compress_detail(
        T*             uncompressed,
        double const   eb,
        int const      radius,
        int const      pardeg,
        uint32_t const codecs_in_use,
        int const      nz_density_factor,
        BYTE*&         compressed,
        size_t&        compressed_len,
        bool           codec_force_fallback,
        cudaStream_t   stream    = nullptr,
        bool           dbg_print = false)
    {
        
        header.codecs_in_use     = codecs_in_use;
        header.nz_density_factor = nz_density_factor;

        T*     d_anchor{nullptr};   // predictor out1
        E*     d_errctrl{nullptr};  // predictor out2
        BYTE*  d_spfmt{nullptr};
        size_t spfmt_out_len{0};

        BYTE*  d_codec_out{nullptr};
        size_t codec_out_len{0};

        size_t data_len, m, errctrl_len, sublen;

            // float entropies[1] = {1.142};
        // float entropies[100] = {1.016,1.075,1.131,1.184,1.236,1.286,1.335,1.382,1.428,1.474,1.518,1.563,1.607,1.651,1.695,1.738,1.782,1.826,1.869,1.913,1.957,2.0,2.044,2.088,2.131,2.175,2.219,2.262,2.306,2.35,2.393,2.437,2.481,2.524,2.568,2.611,2.655,2.699,2.742,2.786,2.83,2.873,2.917,2.961,3.004,3.048,3.092,3.135,3.179,3.223,3.266,3.31,3.354,3.397,3.441,3.484,3.528,3.572,3.615,3.659,3.703,3.746,3.79,3.834,3.877,3.921,3.965,4.008,4.052,4.096,4.139,4.183,4.227,4.27,4.314,4.357,4.401,4.445,4.488,4.532,4.576,4.619,4.663,4.707,4.75,4.794,4.838,4.881,4.925,4.969,5.012,5.056,5.099,5.143,5.187,5.23,5.274,5.318,5.361,5.405
        // };

        /** Cauchy entropies **/
        // float entropies[100] = {0.32,0.345,0.372,0.401,0.432,0.465,0.5,0.537,0.577,0.62,0.664,0.712,0.762,0.815,0.871,0.929,0.991,1.055,1.122,1.191,1.264,1.339,1.416,1.495,1.577,1.66,1.745,1.831,1.918,2.005,2.094,2.182,2.271,2.359,2.446,2.533,2.619,2.703,2.786,2.868,2.949,3.028,3.106,3.182,3.257,3.33,3.403,3.474,3.545,3.615,3.683,3.752,3.819,3.887,3.954,4.02,4.087,4.153,4.219,4.285,4.35,4.416,4.482,4.547,4.612,4.678,4.743,4.808,4.873,4.938,5.003,5.068,5.132,5.197,5.262,5.326,5.39,5.455,5.519,5.583,5.647,5.71,5.774,5.838,5.901,5.964,6.027,6.09,6.153,6.216,6.278,6.34,6.402,6.464,6.526,6.588,6.649,6.71,6.771,6.832};
        
        /** Laplace entropies **/
        float entropies[100] = {0.002,0.003,0.005,0.006,0.009,0.012,0.016,0.021,0.027,0.036,0.046,0.058,0.072,0.089,0.109,0.132,0.158,0.188,0.221,0.258,0.299,0.343,0.391,0.442,0.497,0.555,0.616,0.68,0.746,0.815,0.885,0.958,1.032,1.107,1.183,1.261,1.338,1.416,1.495,1.573,1.652,1.73,1.808,1.886,1.963,2.04,2.117,2.193,2.269,2.344,2.419,2.493,2.567,2.64,2.713,2.786,2.858,2.93,3.001,3.072,3.143,3.214,3.284,3.354,3.423,3.493,3.562,3.631,3.7,3.769,3.838,3.907,3.975,4.043,4.111,4.18,4.248,4.315,4.383,4.451,4.519,4.587,4.654,4.722,4.789,4.857,4.924,4.992,5.059,5.126,5.194,5.261,5.328,5.396,5.463,5.53,5.597,5.665,5.732,5.799};
        
        // must precede the following derived lengths
        int index;

        FILE *ind_file = fopen("entropy_select.txt","r");
        
        fscanf(ind_file, "%d", &index);

        fclose(ind_file);
        float entropy = entropies[index];
    
    


        auto predictor_do = [&]() {
            (*predictor).construct(data_len3, uncompressed, d_anchor, d_errctrl, eb, radius, stream);
            // size_t quant_len = (*predictor).get_len_quant();
            // int* quant_codes = (int*) malloc(sizeof(int)*quant_len);
            // cudaMemcpy(quant_codes, d_errctrl, sizeof(int)*quant_len,cudaMemcpyDeviceToHost);

            // FILE *q_file = fopen("quants.data","wb");
            // fwrite(quant_codes, sizeof(int), quant_len, q_file);
            // fclose(q_file);
            // free(quant_codes);
        };

        auto spcodec_do = [&]() { (*spcodec).encode(uncompressed, m * m, d_spfmt, spfmt_out_len, stream, dbg_print); };

        auto codec_do_with_exception = [&]() {
            auto encode_with_fallback_codec = [&]() {
                use_fallback_codec = true;
                if (not fallback_codec_allocated) {
                    LOGGING(LOG_EXCEPTION, "online allocate fallback (8-byte) codec");

                    (*fb_codec).init(errctrl_len, radius * 2, pardeg, /*dbg print*/ false);
                    fallback_codec_allocated = true;
                }
                (*fb_codec).encode(
                    d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream, true, entropy);
            };

            if (not codec_force_fallback) {
                try {
                    (*codec).encode(
                        d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream, true, entropy);
                }
                catch (const std::runtime_error& e) {
                    LOGGING(LOG_EXCEPTION, "switch to fallback codec");
                    encode_with_fallback_codec();
                }
            }
            else {
                LOGGING(LOG_INFO, "force switch to fallback codec");

                encode_with_fallback_codec();
            }
        };

        auto update_header = [&]() {
            header.x          = data_len3.x;
            header.y          = data_len3.y;
            header.z          = data_len3.z;
            header.radius     = radius;
            header.vle_pardeg = pardeg;
            header.eb         = eb;
            header.byte_vle   = use_fallback_codec ? 8 : 4;
        };

        auto subfile_collect = [&]() {
            header.header_nbyte = sizeof(HEADER);
            uint32_t nbyte[HEADER::END];
            nbyte[HEADER::HEADER] = 128;
            nbyte[HEADER::ANCHOR] = sizeof(T) * (*predictor).get_len_anchor();
            nbyte[HEADER::VLE]    = sizeof(BYTE) * codec_out_len;
            nbyte[HEADER::SPFMT]  = sizeof(BYTE) * spfmt_out_len;

            header.entry[0] = 0;
            // *.END + 1; need to know the ending position
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

            auto debug_header_entry = [&]() {
                printf("\nsubfile collect in compressor:\n");
                printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)HEADER::VAR, 14, #VAR, header.entry[HEADER::VAR]);
                PRINT_ENTRY(HEADER);
                PRINT_ENTRY(ANCHOR);
                PRINT_ENTRY(VLE);
                PRINT_ENTRY(SPFMT);
                PRINT_ENTRY(END);
                printf("\n");
#undef PRINT_ENTRY
            };

            if (dbg_print) debug_header_entry();

            CHECK_CUDA(cudaMemcpyAsync(d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

            D2D_CPY(anchor, ANCHOR)
            D2D_CPY(codec_out, VLE)
            D2D_CPY(spfmt, SPFMT)

            /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        // execution below
        // ---------------

        predictor_do();

        data_len    = (*predictor).get_len_data();
        m           = Reinterpret1DTo2D::get_square_size(data_len);
        errctrl_len = (*predictor).get_len_quant();
        sublen      = ConfigHelper::get_npart(data_len, pardeg);

        spcodec_do(), codec_do_with_exception();

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        update_header(), subfile_collect();
        // output
        compressed_len = header.get_filesize();
        compressed     = d_reserved_compressed;

        printf("Final len: %ld\n", compressed_len);
        collect_compress_timerecord();

        // considering that codec can be consecutively in use, and can compress data of different huff-byte
        use_fallback_codec = false;
    
    }

    void clear_buffer()
    {  //
        (*predictor).clear_buffer();
        (*codec).clear_buffer();
        (*spcodec).clear_buffer();
    }

    /**
     * @brief High-level decompress method for this compressor
     *
     * @param header header on host; if null, copy from device binary (from the beginning)
     * @param in_compressed device pointer, the cusz archive bianry
     * @param out_decompressed device pointer, output decompressed data
     * @param stream CUDA stream
     * @param rpt_print control over printing time
     */
    void decompress(
        cuszHEADER*  header,
        BYTE*        in_compressed,
        T*           out_decompressed,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = true)
    {
        // TODO host having copy of header when compressing
        if (not header) {
            header = new HEADER;
            CHECK_CUDA(cudaMemcpyAsync(header, in_compressed, sizeof(HEADER), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }

        data_len3 = dim3(header->x, header->y, header->z);

        use_fallback_codec      = header->byte_vle == 8;
        double const eb         = header->eb;
        int const    radius     = header->radius;
        auto const   vle_pardeg = header->vle_pardeg;

        // The inputs of components are from `compressed`.
        auto d_anchor = ACCESSOR(ANCHOR, T);
        auto d_vle    = ACCESSOR(VLE, BYTE);
        auto d_sp     = ACCESSOR(SPFMT, BYTE);

        // wire the workspace
        auto d_errctrl = (*predictor).expose_quant();  // reuse space

        // wire and aliasing
        auto d_outlier       = out_decompressed;
        auto d_outlier_xdata = out_decompressed;

        auto spcodec_do              = [&]() { (*spcodec).decode(d_sp, d_outlier, stream); };
        auto codec_do_with_exception = [&]() {
            if (not use_fallback_codec) {  //
                (*codec).decode(d_vle, d_errctrl);
            }
            else {
                if (not fallback_codec_allocated) {
                    (*fb_codec).init((*predictor).get_len_quant(), radius * 2, vle_pardeg, /*dbg print*/ false);
                    fallback_codec_allocated = true;
                }
                (*fb_codec).decode(d_vle, d_errctrl);
            }
        };
        auto predictor_do = [&]() {
            (*predictor).reconstruct(data_len3, d_outlier_xdata, d_anchor, d_errctrl, eb, radius, stream);
        };

        // process
        spcodec_do(), codec_do_with_exception(), predictor_do();

        collect_decompress_timerecord();

        // clear state for the next decompression after reporting
        use_fallback_codec = false;
    }
};

template <typename InputData = float>
struct Framework {
    using DATA    = InputData;
    using ERRCTRL = ErrCtrlTrait<2>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using LorenzoFeatured = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>,
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>  //
        >;

    using Spline3Featured = PredictorReducerCodecBinding<
        cusz::Spline3<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>,
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>  //
        >;

    using DefaultCompressor         = class Compressor<LorenzoFeatured>;
    using LorenzoFeaturedCompressor = class Compressor<LorenzoFeatured>;
    using Spline3FeaturedCompressor = class Compressor<Spline3Featured>;
};

}  // namespace cusz

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef D2D_CPY
#undef ACCESSOR
#undef COLLECT_TIME

#endif
