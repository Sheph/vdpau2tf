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

const int net_input_w = 128;
const int net_input_h = 72;
const int net_grid_w = 16;
const int net_grid_h = 9;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

class VideoDecoder
{
public:
	VideoDecoder(const std::string& filename, int dest_width, int dest_height)
	: dest_width_(dest_width),
	  dest_height_(dest_height),
	  demuxer_(filename.c_str()),
	  dest_frame_ptr_(0),
	  dest_frame_allocated_(false),
	  num_frames_in_use_(0)
	{
		static const int max_decode_surfaces = 20;

		CUVIDPARSERPARAMS videoParserParameters = {};
		videoParserParameters.CodecType = FFmpeg2NvCodecId(demuxer_.GetVideoCodec());
		videoParserParameters.ulMaxNumDecodeSurfaces = max_decode_surfaces;
		videoParserParameters.ulMaxDisplayDelay = 0;
		videoParserParameters.pUserData = this;
		videoParserParameters.pfnSequenceCallback = handle_video_sequence;
		videoParserParameters.pfnDecodePicture = handle_picture_decode;
		videoParserParameters.pfnDisplayPicture = handle_picture_display;

		CUresult res = cuvidCreateVideoParser(&parser_, &videoParserParameters);
		MY_CHECK(res == CUDA_SUCCESS);

		frame_in_use_.resize(max_decode_surfaces);

		thr_ = boost::thread(boost::bind(&VideoDecoder::decode_thread_fn, this));
	}

	~VideoDecoder()
	{
		thr_.join();
	}

	CUdeviceptr get_frame_ptr()
	{
		do {
			if (dest_frame_allocated_) {
				boost::mutex::scoped_lock lock(mtx_);
				return dest_frame_ptr_;
			} else {
				usleep(1000);
			}
		} while (true);
	}

	bool get_next_frame()
	{
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
	}

	static int CUDAAPI handle_video_sequence(void *pUserData, CUVIDEOFORMAT *pVideoFormat)
	{
		std::cout << "SEQ\n";
	}

	static int CUDAAPI handle_picture_decode(void *pUserData, CUVIDPICPARAMS *pPicParams)
	{
		std::cout << "DEC\n";

		VideoDecoder* this_ = (VideoDecoder*)pUserData;

		boost::mutex::scoped_lock lock(this_->mtx_);
		while (this_->num_frames_in_use_ >= this_->frame_in_use_.size()) {
			this_->cond_.wait(lock);
		}
	}

	static int CUDAAPI handle_picture_display(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
	{
		std::cout << "DPY\n";
	}

	int dest_width_;
	int dest_height_;

	FFmpegDemuxer demuxer_;

	CUvideoparser parser_;

	boost::thread thr_;
	boost::mutex mtx_;
	boost::condition_variable cond_;
	CUvideodecoder decoder_;
	CUdeviceptr dest_frame_ptr_;
	bool dest_frame_allocated_;
	std::vector<uint8_t> frame_in_use_;
	size_t num_frames_in_use_;
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

	return 0;
}
