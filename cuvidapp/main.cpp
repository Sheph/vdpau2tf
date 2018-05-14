extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/vdpau.h>
}

#include <cuda.h>
#include "nvcuvid.h"
#include "FFmpegDemuxer.h"

#undef LOG

#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/allocator_registry.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/grappler/devices.h>
#include <tensorflow/core/common_runtime/gpu/gpu_init.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/common_runtime/gpu_device_context.h>
#include <tensorflow/stream_executor/stream_executor_pimpl.h>
#include <tensorflow/stream_executor/cuda/cuda_gpu_executor.h>
#include <tensorflow/stream_executor/cuda/cuda_stream.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <malloc.h>
#include <unistd.h>
#include <sys/time.h>

extern "C" {
extern void __assert_fail (const char *__assertion, const char *__file,
	unsigned int __line, const char *__function) __THROW __attribute__ ((__noreturn__));
}

#define MY_CHECK(expr) if (!(expr)) __assert_fail(__STRING(expr), __FILE__, __LINE__, __PRETTY_FUNCTION__)

static int gpu_id = -1;
static CUdevice cuda_device = 0;
static CUcontext cuda_ctx = NULL;

static const int net_input_w = 128;
static const int net_input_h = 72;
static const int net_grid_w = 16;
static const int net_grid_h = 9;

static const char * GetVideoCodecString(cudaVideoCodec eCodec) {
	static struct {
		cudaVideoCodec eCodec;
		const char *name;
	} aCodecName [] = {
		{ cudaVideoCodec_MPEG1,     "MPEG-1"       },
		{ cudaVideoCodec_MPEG2,     "MPEG-2"       },
		{ cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
		{ cudaVideoCodec_VC1,       "VC-1/WMV"     },
		{ cudaVideoCodec_H264,      "AVC/H.264"    },
		{ cudaVideoCodec_JPEG,      "M-JPEG"       },
		{ cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
		{ cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
		{ cudaVideoCodec_HEVC,      "H.265/HEVC"   },
		{ cudaVideoCodec_VP8,       "VP8"          },
		{ cudaVideoCodec_VP9,       "VP9"          },
		{ cudaVideoCodec_NumCodecs, "Invalid"      },
		{ cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
		{ cudaVideoCodec_YV12,      "YV12 4:2:0"   },
		{ cudaVideoCodec_NV12,      "NV12 4:2:0"   },
		{ cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
		{ cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
	};

	if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
		return aCodecName[eCodec].name;
	}
	for (size_t i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
		if (eCodec == aCodecName[i].eCodec) {
			return aCodecName[eCodec].name;
		}
	}
	return "Unknown";
}

static const char * GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
	static struct {
		cudaVideoChromaFormat eChromaFormat;
		const char *name;
	} aChromaFormatName[] = {
		{ cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
		{ cudaVideoChromaFormat_420,        "YUV 420"              },
		{ cudaVideoChromaFormat_422,        "YUV 422"              },
		{ cudaVideoChromaFormat_444,        "YUV 444"              },
	};

	if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
		return aChromaFormatName[eChromaFormat].name;
	}
	return "Unknown";
}

static unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight) {
	if (eCodec == cudaVideoCodec_VP9) {
		return 12;
	}

	if (eCodec == cudaVideoCodec_H264 || eCodec == cudaVideoCodec_H264_SVC || eCodec == cudaVideoCodec_H264_MVC) {
		// assume worst-case of 20 decode surfaces for H264
		return 20;
	}

	if (eCodec == cudaVideoCodec_HEVC) {
		// ref HEVC spec: A.4.1 General tier and level limits
		// currently assuming level 6.2, 8Kx4K
		int MaxLumaPS = 35651584;
		int MaxDpbPicBuf = 6;
		int PicSizeInSamplesY = (int)(nWidth * nHeight);
		int MaxDpbSize;
		if (PicSizeInSamplesY <= (MaxLumaPS>>2))
			MaxDpbSize = MaxDpbPicBuf * 4;
		else if (PicSizeInSamplesY <= (MaxLumaPS>>1))
			MaxDpbSize = MaxDpbPicBuf * 2;
		else if (PicSizeInSamplesY <= ((3*MaxLumaPS)>>2))
			MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
		else
			MaxDpbSize = MaxDpbPicBuf;
		return (std::min)(MaxDpbSize, 16) + 4;
	}

	return 8;
}

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

class VideoDecoder
{
public:
	VideoDecoder(const std::string& filename, int dest_width, int dest_height)
	: dest_width_(dest_width),
	  dest_height_(dest_height),
	  demuxer_(filename.c_str()),
	  max_decode_surfaces_(0),
	  bit_depth_minus_8_(0),
	  decoder_(0),
	  done_(false)
	{
		CUresult res = cuvidCtxLockCreate(&ctx_lock_, cuda_ctx);
		MY_CHECK(res == CUDA_SUCCESS);

		CUVIDPARSERPARAMS videoParserParameters = {};
		videoParserParameters.CodecType = FFmpeg2NvCodecId(demuxer_.GetVideoCodec());
		videoParserParameters.ulMaxNumDecodeSurfaces = 1;
		videoParserParameters.ulMaxDisplayDelay = 0;
		videoParserParameters.pUserData = this;
		videoParserParameters.pfnSequenceCallback = handle_video_sequence;
		videoParserParameters.pfnDecodePicture = handle_picture_decode;
		videoParserParameters.pfnDisplayPicture = handle_picture_display;

		res = cuvidCreateVideoParser(&parser_, &videoParserParameters);
		MY_CHECK(res == CUDA_SUCCESS);

		thr_ = boost::thread(boost::bind(&VideoDecoder::decode_thread_fn, this));
	}

	~VideoDecoder()
	{
		thr_.join();
	}

	CUdeviceptr get_next_frame(CUstream stream, unsigned int& pitch)
	{
		CUVIDPARSERDISPINFO* disp_info = NULL;

		{
			boost::mutex::scoped_lock lock(mtx_);
			while (frame_queue_.empty() && !done_) {
				cond_.wait(lock);
			}
			if (done_) {
				return 0;
			}
			disp_info = frame_queue_.front();
		}

		CUVIDPROCPARAMS videoProcessingParameters = {};
		videoProcessingParameters.progressive_frame = disp_info->progressive_frame;
		videoProcessingParameters.second_field = disp_info->repeat_first_field + 1;
		videoProcessingParameters.top_field_first = disp_info->top_field_first;
		videoProcessingParameters.unpaired_field = disp_info->repeat_first_field < 0;
		videoProcessingParameters.output_stream = stream;

		CUdeviceptr frame = 0;

		CUresult res = cuvidMapVideoFrame(decoder_, disp_info->picture_index, &frame,
			&pitch, &videoProcessingParameters);
		MY_CHECK(res == CUDA_SUCCESS);

		return frame;
	}

	void finish_frame(CUdeviceptr frame)
	{
		CUresult res = cuvidUnmapVideoFrame(decoder_, frame);
		MY_CHECK(res == CUDA_SUCCESS);

		{
			boost::mutex::scoped_lock lock(mtx_);
			frame_queue_.pop_front();
			cond_.notify_all();
		}
	}

	cv::Mat device_frame_to_mat(CUdeviceptr frame, unsigned int pitch, CUstream stream)
	{
		MY_CHECK(bit_depth_minus_8_ == 0);

		cv::Mat gray(dest_height_, dest_width_, CV_8UC1);

		CUDA_MEMCPY2D m = { 0 };
		m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		m.srcDevice = frame;
		m.srcPitch = pitch;
		m.dstMemoryType = CU_MEMORYTYPE_HOST;
		m.dstDevice = (CUdeviceptr)(m.dstHost = gray.data);
		m.dstPitch = dest_width_;
		m.WidthInBytes = dest_width_;
		m.Height = dest_height_;
		CUresult res = cuMemcpy2DAsync(&m, stream);
		MY_CHECK(res == CUDA_SUCCESS);
		res = cuStreamSynchronize(stream);
		MY_CHECK(res == CUDA_SUCCESS);

		return gray;
	}

private:
	void decode_thread_fn()
	{
		CUresult res = cuCtxSetCurrent(cuda_ctx);
		MY_CHECK(res == CUDA_SUCCESS);

		int frame_bytes = 0;
		CUVIDSOURCEDATAPACKET packet = {0};
		do {
			uint8_t* frame_ptr = NULL;
			demuxer_.Demux(&frame_ptr, &frame_bytes);

			packet.payload = frame_ptr;
			packet.payload_size = frame_bytes;
			packet.flags = CUVID_PKT_TIMESTAMP;
			packet.timestamp = 0;
			if (!frame_ptr || frame_bytes == 0) {
				packet.flags |= CUVID_PKT_ENDOFSTREAM;
			}
			res = cuvidParseVideoData(parser_, &packet);
			MY_CHECK(res == CUDA_SUCCESS);
		} while (frame_bytes);

		boost::mutex::scoped_lock lock(mtx_);
		done_ = true;
		cond_.notify_all();
	}

	static int CUDAAPI handle_video_sequence(void *pUserData, CUVIDEOFORMAT *pVideoFormat)
	{
		VideoDecoder* this_ = (VideoDecoder*)pUserData;

		if (this_->decoder_) {
			return 1;
		}

		this_->video_info_ << "Video Input Information" << std::endl
			<< "\tCodec        : " << GetVideoCodecString(pVideoFormat->codec) << std::endl
			<< "\tFrame rate   : " << pVideoFormat->frame_rate.numerator << "/" << pVideoFormat->frame_rate.denominator
			<< " = " << 1.0 * pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator << " fps" << std::endl
			<< "\tSequence     : " << (pVideoFormat->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
			<< "\tCoded size   : [" << pVideoFormat->coded_width << ", " << pVideoFormat->coded_height << "]" << std::endl
			<< "\tDisplay area : [" << pVideoFormat->display_area.left << ", " << pVideoFormat->display_area.top << ", "
			<< pVideoFormat->display_area.right << ", " << pVideoFormat->display_area.bottom << "]" << std::endl
			<< "\tChroma       : " << GetVideoChromaFormatString(pVideoFormat->chroma_format) << std::endl
			<< "\tBit depth    : " << pVideoFormat->bit_depth_luma_minus8 + 8
		;
		this_->video_info_ << std::endl;

		this_->max_decode_surfaces_ = GetNumDecodeSurfaces(pVideoFormat->codec, pVideoFormat->coded_width, pVideoFormat->coded_height);

		CUVIDDECODECAPS decodecaps;
		memset(&decodecaps, 0, sizeof(decodecaps));

		decodecaps.eCodecType = pVideoFormat->codec;
		decodecaps.eChromaFormat = pVideoFormat->chroma_format;
		decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

		CUresult res = cuvidGetDecoderCaps(&decodecaps);
		MY_CHECK(res == CUDA_SUCCESS);

		if (!decodecaps.bIsSupported) {
			std::cout << "Codec not supported on this GPU\n";
			return this_->max_decode_surfaces_;
		}

		if ((pVideoFormat->coded_width > decodecaps.nMaxWidth) ||
			(pVideoFormat->coded_height > decodecaps.nMaxHeight)) {
			std::ostringstream errorString;
			errorString << std::endl
						<< "Resolution          : " << pVideoFormat->coded_width << "x" << pVideoFormat->coded_height << std::endl
						<< "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x" << decodecaps.nMaxHeight << std::endl
						<< "Resolution not supported on this GPU";
			const std::string cErr = errorString.str();
			std::cout << cErr << "\n";
			return this_->max_decode_surfaces_;
		}

		if ((pVideoFormat->coded_width>>4)*(pVideoFormat->coded_height>>4) > decodecaps.nMaxMBCount) {
			std::ostringstream errorString;
			errorString << std::endl
						<< "MBCount             : " << (pVideoFormat->coded_width >> 4)*(pVideoFormat->coded_height >> 4) << std::endl
						<< "Max Supported mbcnt : " << decodecaps.nMaxMBCount << std::endl
						<< "MBCount not supported on this GPU";
			const std::string cErr = errorString.str();
			std::cout << cErr << "\n";
			return this_->max_decode_surfaces_;
		}

		// eCodec has been set in the constructor (for parser). Here it's set again for potential correction
		this_->bit_depth_minus_8_ = pVideoFormat->bit_depth_luma_minus8;

		CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
		videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
		videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
		videoDecodeCreateInfo.OutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
		videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
		videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
		videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
		// With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
		videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
		videoDecodeCreateInfo.ulNumDecodeSurfaces = this_->max_decode_surfaces_;
		videoDecodeCreateInfo.vidLock = this_->ctx_lock_;
		videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
		videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;

		videoDecodeCreateInfo.display_area.left = pVideoFormat->display_area.left;
		videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
		videoDecodeCreateInfo.display_area.right = pVideoFormat->display_area.right;
		videoDecodeCreateInfo.display_area.bottom = pVideoFormat->display_area.bottom;
		videoDecodeCreateInfo.ulTargetWidth = this_->dest_width_;
		videoDecodeCreateInfo.ulTargetHeight = this_->dest_height_;

		this_->video_info_ << "Video Decoding Params:" << std::endl
			<< "\tNum Surfaces : " << videoDecodeCreateInfo.ulNumDecodeSurfaces << std::endl
			<< "\tCrop         : [" << videoDecodeCreateInfo.display_area.left << ", " << videoDecodeCreateInfo.display_area.top << ", "
			<< videoDecodeCreateInfo.display_area.right << ", " << videoDecodeCreateInfo.display_area.bottom << "]" << std::endl
			<< "\tResize       : " << videoDecodeCreateInfo.ulTargetWidth << "x" << videoDecodeCreateInfo.ulTargetHeight << std::endl
			<< "\tDeinterlace  : " << std::vector<const char *>{"Weave", "Bob", "Adaptive"}[videoDecodeCreateInfo.DeinterlaceMode]
		;
		this_->video_info_ << std::endl;

		res = cuvidCreateDecoder(&this_->decoder_, &videoDecodeCreateInfo);
		MY_CHECK(res == CUDA_SUCCESS);

		std::cout << this_->video_info_.str() << std::endl;

		return this_->max_decode_surfaces_;
	}

	static int CUDAAPI handle_picture_decode(void *pUserData, CUVIDPICPARAMS *pPicParams)
	{
		VideoDecoder* this_ = (VideoDecoder*)pUserData;

		{
			boost::mutex::scoped_lock lock(this_->mtx_);
			while (static_cast<int>(this_->frame_queue_.size()) >= this_->max_decode_surfaces_) {
				this_->cond_.wait(lock);
			}
		}

		CUresult res = cuvidDecodePicture(this_->decoder_, pPicParams);
		MY_CHECK(res == CUDA_SUCCESS);

		return 1;
	}

	static int CUDAAPI handle_picture_display(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
	{
		VideoDecoder* this_ = (VideoDecoder*)pUserData;

		MY_CHECK(pDispInfo->picture_index < this_->max_decode_surfaces_);

		boost::mutex::scoped_lock lock(this_->mtx_);
		this_->frame_queue_.push_back(pDispInfo);
		this_->cond_.notify_all();

		return 1;
	}

	int dest_width_;
	int dest_height_;

	FFmpegDemuxer demuxer_;

	CUvideoctxlock ctx_lock_;

	CUvideoparser parser_;

	boost::thread thr_;
	boost::mutex mtx_;
	boost::condition_variable cond_;
	std::ostringstream video_info_;
	int max_decode_surfaces_;
	int bit_depth_minus_8_;
	CUvideodecoder decoder_;
	std::list<CUVIDPARSERDISPINFO*> frame_queue_;
	bool done_;
};

int main(int argc, char* argv[])
{
	if (argc <= 2) {
		printf("cuvidapp [gpu_id] [file]\n");
		return 0;
	}

	gpu_id = atoi(argv[1]);

	cuInit(0);

	CUresult res = cuDeviceGet(&cuda_device, gpu_id);
	MY_CHECK(res == CUDA_SUCCESS);

	res = cuCtxCreate(&cuda_ctx, CU_CTX_MAP_HOST, cuda_device);
	MY_CHECK(res == CUDA_SUCCESS);

	VideoDecoder decoder(argv[2], net_input_w, net_input_h);

	struct timeval tv1;
	gettimeofday(&tv1, NULL);

	int num_frames = 0;

	CUdeviceptr frame = 0;
	unsigned int pitch = 0;

	while ((frame = decoder.get_next_frame(0, pitch))) {
		cv::Mat gray = decoder.device_frame_to_mat(frame, pitch, 0);
		cv::imshow("frame", gray);
		cv::waitKey(1);
		decoder.finish_frame(frame);
		++num_frames;
	}

	struct timeval tv2, tv_res;
	gettimeofday(&tv2, NULL);

	timersub(&tv2, &tv1, &tv_res);

	double tm = (double)tv_res.tv_sec + (double)tv_res.tv_usec / 1000000.0;

	printf("DONE %f fps!\n", (float)(num_frames / tm));

	fflush(stdout);

	return 0;
}
