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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/vdpau.h>
}

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/XKBlib.h>
#include <X11/XF86keysym.h>
#define XK_MISCELLANY
#define XK_LATIN1
#define XK_XKB_KEYS
#include <X11/keysymdef.h>
#include <X11/extensions/xf86vmode.h>
#include <vdpau/vdpau_x11.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cudaVDPAU.h>

#undef Status

#define OUT_WIDTH 128
#define OUT_HEIGHT 72

class VDPAUAllocator : public tensorflow::Allocator
{
public:
	explicit VDPAUAllocator(void* vdpau_ptr)
	: vdpau_ptr(vdpau_ptr)
	{
	}
	virtual ~VDPAUAllocator() { }

	tensorflow::string Name() override { return "vdpau"; }

	void* AllocateRaw(size_t alignment, size_t num_bytes) override
	{
		return vdpau_ptr;
	}

	void DeallocateRaw(void* ptr) override { }

private:
	void* vdpau_ptr;
};

struct VideoBuffer
{
	VdpVideoSurface video_surface;
	bool busy = false;
};

static Display* dpy = NULL;
static Window win = 0;
static GC gc = 0;

static VdpOutputSurface output_surface;
static VideoBuffer video_buffers[3];
static VdpProcamp procamp;
static VdpVideoMixer video_mixer;
static uint32_t vid_width, vid_height;
static VdpChromaType vdp_chroma_type;
static VdpDevice vdp_device;
static int colorspace;
static VdpPresentationQueueTarget vdp_target;
static VdpPresentationQueue vdp_queue;
static VdpDecoder vdp_decoder;

static VdpGetProcAddress *vdp_get_proc_address;
static VdpDeviceDestroy *vdp_device_destroy;
static VdpVideoSurfaceCreate *vdp_video_surface_create;
static VdpVideoSurfaceDestroy *vdp_video_surface_destroy;
static VdpVideoSurfacePutBitsYCbCr *vdp_video_surface_put_bits_y_cb_cr;
static VdpOutputSurfaceGetBitsNative *vdp_output_surface_get_bits_native;
static VdpOutputSurfaceCreate *vdp_output_surface_create;
static VdpOutputSurfaceDestroy *vdp_output_surface_destroy;
static VdpVideoMixerCreate *vdp_video_mixer_create;
static VdpVideoMixerDestroy *vdp_video_mixer_destroy;
static VdpVideoMixerRender *vdp_video_mixer_render;
static VdpVideoMixerSetFeatureEnables *vdp_video_mixer_set_feature_enables;
static VdpVideoMixerSetAttributeValues *vdp_video_mixer_set_attribute_values;
static VdpPresentationQueueTargetDestroy *vdp_presentation_queue_target_destroy;
static VdpPresentationQueueCreate *vdp_presentation_queue_create;
static VdpPresentationQueueDestroy *vdp_presentation_queue_destroy;
static VdpPresentationQueueDisplay *vdp_presentation_queue_display;
static VdpPresentationQueueTargetCreateX11 *vdp_presentation_queue_target_create_x11;
static VdpDecoderCreate *vdp_decoder_create;
static VdpDecoderDestroy *vdp_decoder_destroy;
static VdpDecoderRender *vdp_decoder_render;
static VdpDecoderQueryCapabilities *vdp_decoder_query_capabilities;
static VdpDecoderGetParameters *vdp_decoder_get_parameters;
static VdpGenerateCSCMatrix *vdp_generate_csc_matrix;

static enum AVPixelFormat get_format_vdpau(struct AVCodecContext *avctx,
										   const enum AVPixelFormat *valid_fmts);
static int get_buffer_vdpau(struct AVCodecContext *c, AVFrame *pic, int flags);

class CompressedVideoStream
{
public:
	explicit CompressedVideoStream(const std::string& filename)
	{
		av_register_all();
		avcodec_register_all();
		avformat_network_init();

		int r;
		AVDictionary* dict = 0;
		r = avformat_open_input(&format_, filename.c_str(), 0, &dict);
		assert(format_);

		r = avformat_find_stream_info(format_, NULL);
		assert(r >= 0);

		for (int c = 0; c < (int)format_->nb_streams; c++) {
			AVCodecContext* cx = format_->streams[c]->codec;
			if (cx->codec_type==AVMEDIA_TYPE_VIDEO) {
				assert(cx->codec_id != AV_CODEC_ID_NONE && cx->codec_id != AV_CODEC_ID_RAWVIDEO);
				cx_video_ = format_->streams[c]->codec;
				video_stream_index_ = c;
				break;
			}
		}

		assert(video_stream_index_ != -1);

		AVCodec* tmp_dec = avcodec_find_decoder(format_->streams[video_stream_index_]->codecpar->codec_id);
		assert(tmp_dec);

		cx_video_hw_ = avcodec_alloc_context3(tmp_dec);
		r = avcodec_parameters_to_context(cx_video_hw_, format_->streams[video_stream_index_]->codecpar);
		assert(r >= 0);
		av_codec_set_pkt_timebase(cx_video_hw_, format_->streams[video_stream_index_]->time_base);

		cx_video_hw_->opaque = (void*)this;
		cx_video_hw_->get_buffer2 = get_buffer_vdpau;
		cx_video_hw_->get_format = get_format_vdpau;
		cx_video_hw_->slice_flags = SLICE_FLAG_CODED_ORDER | SLICE_FLAG_ALLOW_FIELD;

		cx_video_hw_->err_recognition = AV_EF_COMPLIANT;
		cx_video_hw_->workaround_bugs = FF_BUG_AUTODETECT;
		cx_video_hw_->error_concealment = FF_EC_GUESS_MVS | FF_EC_DEBLOCK;
		cx_video_hw_->idct_algo = FF_IDCT_AUTO;
		cx_video_hw_->debug = 0;

		vdpau_context_ = av_vdpau_alloc_context();
		memset(vdpau_context_, 0, sizeof(AVVDPAUContext));

		width_ = cx_video_->width;
		height_ = cx_video_->height;

		r = avcodec_open2(cx_video_hw_, tmp_dec, NULL);
		assert(r >= 0);
	}

	~CompressedVideoStream()
	{
		avformat_close_input(&format_);
	}

	bool get_next_frame()
	{
		AVPacket pkt;

		int ret = av_read_frame(format_, &pkt);
		if (ret < 0) {
			return false;
		}

		if (pkt.stream_index != video_stream_index_) {
			return true;
		}

		while (true) {
			bool resend_packet = false;
			int ret = avcodec_send_packet(cx_video_hw_, &pkt);
			if (ret < 0 && ret == AVERROR(EAGAIN))
				resend_packet = true;
			else if (ret < 0)
				break;

			AVFrame* frame = av_frame_alloc();

			ret = avcodec_receive_frame(cx_video_hw_, frame);

			if (ret == AVERROR(EAGAIN) || resend_packet) {
				av_frame_unref(frame);
				continue;
			} else if (ret == 0) {
				av_frame_unref(frame);
				break;
			} else if (ret < 0) {
				printf("DAMN!\n");
				av_frame_unref(frame);
				break;
			}
		}

		av_packet_unref(&pkt);

		return true;
	}

	int get_width() const { return width_; }
	int get_height() const { return height_; }

	AVFormatContext* format_ = 0;
	AVCodecContext* cx_video_ = 0;
	int video_stream_index_ = -1;
	AVCodecContext* cx_video_hw_;
	AVVDPAUContext* vdpau_context_;
	AVFrame* frame_;

	int width_ = 0;
	int height_ = 0;
};

static int render_vdpau(struct AVCodecContext *s, AVFrame *src,
	const VdpPictureInfo *info,
	uint32_t count,
	const VdpBitstreamBuffer *buffers)
{
	if (!src)
		return -1;

	VideoBuffer* buff = (VideoBuffer*)src->opaque;

	VdpStatus vdp_st = vdp_decoder_render(vdp_decoder, buff->video_surface, info, count, buffers);
	assert(vdp_st == VDP_STATUS_OK);

	VdpVideoMixerPictureStructure field = VDP_VIDEO_MIXER_PICTURE_STRUCTURE_FRAME;

	vdp_video_mixer_render(video_mixer, VDP_INVALID_HANDLE, 0,
		field, 0, (VdpVideoSurface*)VDP_INVALID_HANDLE,
		buff->video_surface,
		0, (VdpVideoSurface*)VDP_INVALID_HANDLE,
		NULL,
		output_surface,
		NULL, NULL, 0, NULL);

	return 0;
}

static enum AVPixelFormat get_format_vdpau(struct AVCodecContext *avctx,
										   const enum AVPixelFormat *valid_fmts)
{
	CompressedVideoStream* this_ = (CompressedVideoStream*)avctx->opaque;
	avctx->hwaccel_context = this_->vdpau_context_;
	int r = av_vdpau_bind_context(avctx, vdp_device, vdp_get_proc_address, 0);
	assert(r == 0);
	((AVVDPAUContext*)(avctx->hwaccel_context))->render2 = render_vdpau;

	while (*valid_fmts != AV_PIX_FMT_NONE) {
		if (avctx->hwaccel_context and (*valid_fmts == AV_PIX_FMT_VDPAU))
			return AV_PIX_FMT_VDPAU;
		if (!avctx->hwaccel_context and (*valid_fmts == AV_PIX_FMT_YUV420P))
			return AV_PIX_FMT_YUV420P;
		valid_fmts++;
	}
	return AV_PIX_FMT_NONE;
}

static void release_buffer_vdpau(void *opaque, uint8_t *data)
{
	VideoBuffer* buff = (VideoBuffer*)data;
	buff->busy = false;
}

static int get_buffer_vdpau(struct AVCodecContext *c, AVFrame *pic, int /*flags*/)
{
	int buff_i = -1;
	for (size_t i = 0; i < sizeof(video_buffers)/sizeof(video_buffers[0]); ++i) {
		if (!video_buffers[i].busy) {
			buff_i = i;
			video_buffers[i].busy = true;
			break;
		}
	}

	assert(buff_i != -1);

	for (int i = 0; i < 4; i++) {
		pic->data[i] = NULL;
		pic->linesize[i] = 0;
	}
	pic->opaque = &video_buffers[buff_i];
	pic->reordered_opaque = c->reordered_opaque;
	pic->data[3] = (uint8_t*)(uintptr_t)video_buffers[buff_i].video_surface;

	AVBufferRef *buffer =
		av_buffer_create((uint8_t*)&video_buffers[buff_i], 0, release_buffer_vdpau, NULL, 0);
	pic->buf[0] = buffer;

	return 0;
}

static void initX()
{
	unsigned long black, white;

	dpy = XOpenDisplay((char*)0);
	black = BlackPixel(dpy, DefaultScreen(dpy)),
	white = WhitePixel(dpy, DefaultScreen(dpy));
	win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0,
			OUT_WIDTH, OUT_HEIGHT, 5, black, white);
	XSetStandardProperties(dpy, win, "VDPAU Test", "VDPAU",
			None, NULL, 0, NULL);
	XSelectInput(dpy, win, ExposureMask|ButtonPressMask|KeyPressMask);
	gc = XCreateGC(dpy, win, 0, NULL);
	XSetBackground(dpy, gc, white);
	XSetBackground(dpy, gc, black);
	XClearWindow(dpy, win);
}

static void cleanupX()
{
	XFreeGC(dpy, gc);
	XDestroyWindow(dpy, win);
	XCloseDisplay(dpy);
}

static int initVDPAU()
{
	VdpStatus vdp_st;

	struct vdp_function
	{
		const int id;
		void *pointer;
	};

	const struct vdp_function *dsc;

	static const struct vdp_function vdp_func[] = {
		{VDP_FUNC_ID_DEVICE_DESTROY,                    &vdp_device_destroy},
		{VDP_FUNC_ID_VIDEO_SURFACE_CREATE,              &vdp_video_surface_create},
		{VDP_FUNC_ID_VIDEO_SURFACE_DESTROY,             &vdp_video_surface_destroy},
		{VDP_FUNC_ID_VIDEO_SURFACE_PUT_BITS_Y_CB_CR,    &vdp_video_surface_put_bits_y_cb_cr},
		{VDP_FUNC_ID_OUTPUT_SURFACE_GET_BITS_NATIVE,    &vdp_output_surface_get_bits_native},
		{VDP_FUNC_ID_OUTPUT_SURFACE_CREATE,             &vdp_output_surface_create},
		{VDP_FUNC_ID_OUTPUT_SURFACE_DESTROY,            &vdp_output_surface_destroy},
		{VDP_FUNC_ID_VIDEO_MIXER_CREATE,                &vdp_video_mixer_create},
		{VDP_FUNC_ID_VIDEO_MIXER_DESTROY,               &vdp_video_mixer_destroy},
		{VDP_FUNC_ID_VIDEO_MIXER_RENDER,                &vdp_video_mixer_render},
		{VDP_FUNC_ID_VIDEO_MIXER_SET_FEATURE_ENABLES,   &vdp_video_mixer_set_feature_enables},
		{VDP_FUNC_ID_VIDEO_MIXER_SET_ATTRIBUTE_VALUES,  &vdp_video_mixer_set_attribute_values},
		{VDP_FUNC_ID_PRESENTATION_QUEUE_TARGET_DESTROY, &vdp_presentation_queue_target_destroy},
		{VDP_FUNC_ID_PRESENTATION_QUEUE_CREATE,         &vdp_presentation_queue_create},
		{VDP_FUNC_ID_PRESENTATION_QUEUE_DESTROY,        &vdp_presentation_queue_destroy},
		{VDP_FUNC_ID_PRESENTATION_QUEUE_DISPLAY,        &vdp_presentation_queue_display},
		{VDP_FUNC_ID_PRESENTATION_QUEUE_TARGET_CREATE_X11,
			&vdp_presentation_queue_target_create_x11},
		{VDP_FUNC_ID_DECODER_CREATE,                    &vdp_decoder_create},
		{VDP_FUNC_ID_DECODER_RENDER,                    &vdp_decoder_render},
		{VDP_FUNC_ID_DECODER_DESTROY,                   &vdp_decoder_destroy},
		{VDP_FUNC_ID_DECODER_QUERY_CAPABILITIES,        &vdp_decoder_query_capabilities},
		{VDP_FUNC_ID_DECODER_GET_PARAMETERS,            &vdp_decoder_get_parameters},
		{VDP_FUNC_ID_GENERATE_CSC_MATRIX,               &vdp_generate_csc_matrix},
		{0, NULL}
	};

	vdp_st = vdp_device_create_x11(dpy, DefaultScreen(dpy),
		&vdp_device, &vdp_get_proc_address);

	if (vdp_st != VDP_STATUS_OK) {
		return -1;
	}

	for (dsc = vdp_func; dsc->pointer; dsc++) {
		vdp_st = vdp_get_proc_address(vdp_device, dsc->id, (void**)dsc->pointer);
		if (vdp_st != VDP_STATUS_OK) {
			return -1;
		}
	}

	return 0;
}

static int initVDPAUQueue()
{
	VdpStatus vdp_st;

	vdp_st = vdp_presentation_queue_target_create_x11(vdp_device, win, &vdp_target);
	if (vdp_st != VDP_STATUS_OK)
		return -1;

	XMapRaised(dpy, win);

	vdp_st = vdp_presentation_queue_create(vdp_device, vdp_target, &vdp_queue);
	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

static int updateCSCMatrix()
{
	VdpStatus vdp_st;
	VdpCSCMatrix matrix;
	static const VdpVideoMixerAttribute attributes[] = {VDP_VIDEO_MIXER_ATTRIBUTE_CSC_MATRIX};
	const void *attribute_values[] = {&matrix};
	static const VdpColorStandard vdp_colors[] = {0, VDP_COLOR_STANDARD_ITUR_BT_601, VDP_COLOR_STANDARD_ITUR_BT_709, VDP_COLOR_STANDARD_SMPTE_240M};
	int csp = colorspace;

	if (!csp)
		csp = vid_width >= 1280 || vid_height > 576 ? 2 : 1;

	vdp_st = vdp_generate_csc_matrix(&procamp, vdp_colors[csp], &matrix);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	vdp_st = vdp_video_mixer_set_attribute_values(video_mixer, 1, attributes,
		attribute_values);
	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

static int createVDPMixer(VdpChromaType vdp_chroma_type)
{
#define VDP_NUM_MIXER_PARAMETER 3
#define MAX_NUM_FEATURES 6
	VdpStatus vdp_st;
	int feature_count = 0;

	VdpVideoMixerFeature features[MAX_NUM_FEATURES];

	static const VdpVideoMixerParameter parameters[VDP_NUM_MIXER_PARAMETER] = {
		VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_WIDTH,
		VDP_VIDEO_MIXER_PARAMETER_VIDEO_SURFACE_HEIGHT,
		VDP_VIDEO_MIXER_PARAMETER_CHROMA_TYPE
	};

	const void *const parameter_values[VDP_NUM_MIXER_PARAMETER] = {
		&vid_width,
		&vid_height,
		&vdp_chroma_type
	};

	vdp_st = vdp_video_mixer_create(vdp_device, feature_count, features,
									VDP_NUM_MIXER_PARAMETER,
									parameters, parameter_values,
									&video_mixer);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return updateCSCMatrix();
}

static int createVideoSurface()
{
	VdpStatus vdp_st;
	for (size_t i = 0; i < sizeof(video_buffers)/sizeof(video_buffers[0]); ++i) {
		vdp_st = vdp_video_surface_create(vdp_device, vdp_chroma_type,
			vid_width, vid_height, &video_buffers[i].video_surface);
		assert(vdp_st == VDP_STATUS_OK);
	}

	return 0;
}

static int createOutputSurface()
{
	VdpStatus vdp_st;

	vdp_st = vdp_output_surface_create(vdp_device, VDP_RGBA_FORMAT_B8G8R8A8,
		OUT_WIDTH, OUT_HEIGHT,
		&output_surface);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

static void createDecoder()
{
	VdpStatus vdp_st = vdp_decoder_create(vdp_device, VDP_DECODER_PROFILE_H264_HIGH, vid_width, vid_height, 3, &vdp_decoder);
	assert(vdp_st == VDP_STATUS_OK);
}

static cv::Mat getBits()
{
	VdpStatus vdp_st;

	void* planes[1];
	uint32_t pitches[1] = {OUT_WIDTH * 4};

	cv::Mat data(OUT_HEIGHT, OUT_WIDTH, CV_8UC4);

	planes[0] = data.data;

	vdp_st = vdp_output_surface_get_bits_native(output_surface, NULL,
		planes, pitches);

	if (vdp_st != VDP_STATUS_OK) {
		return cv::Mat();
	}

	return data;
}

static CUcontext vdpau_cuda_ctx = NULL;
static CUgraphicsResource vdpau_cuda_resource = NULL;
static CUdeviceptr vdpau_cuda_data = 0;

static void initVDPAUCuda(int device_id)
{
	cuInit(0);

	CUdevice cuda_device = 0;
	CUresult res = cuDeviceGet(&cuda_device, device_id);
	assert(res == CUDA_SUCCESS);

	res = cuVDPAUCtxCreate(&vdpau_cuda_ctx, CU_CTX_MAP_HOST, cuda_device, vdp_device, vdp_get_proc_address);
	assert(res == CUDA_SUCCESS);

	res = cuGraphicsVDPAURegisterOutputSurface(&vdpau_cuda_resource, output_surface, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	assert(res == CUDA_SUCCESS);

	res = cuMemAlloc(&vdpau_cuda_data, OUT_WIDTH * OUT_HEIGHT * 4);
	assert(res == CUDA_SUCCESS);
}

static boost::shared_ptr<tensorflow::Session> tf_session;
static CUcontext tf_cuda_ctx = NULL;
static CUstream tf_cuda_stream = NULL;

static tensorflow::Status initTF(const std::string& graph_file, int device_id)
{
	tensorflow::GraphDef graph_def;
	TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(
		tensorflow::Env::Default(),
		graph_file,
		&graph_def));

	if (device_id >= 0) {
		std::stringstream ss;
		ss << "/device:GPU:" << device_id;
		tensorflow::graph::SetDefaultDevice(ss.str(), &graph_def);
	}

	tensorflow::SessionOptions opts;
	opts.config.set_allow_soft_placement(true);
	tf_session.reset(tensorflow::NewSession(opts));
	TF_RETURN_IF_ERROR(tf_session->Create(graph_def));

	perftools::gputools::Platform* gpu_manager = tensorflow::GPUMachineManager();
	perftools::gputools::StreamExecutor* se = gpu_manager->ExecutorForDevice(device_id).ValueOrDie();
	perftools::gputools::internal::StreamExecutorInterface* sei	= se->implementation();
	perftools::gputools::cuda::CUDAExecutor* cuda_executor = dynamic_cast<perftools::gputools::cuda::CUDAExecutor*>(sei);

	assert(cuda_executor);

	tensorflow::GPUDeviceContext* dev_ctx = NULL;

	tensorflow::DirectSession* direct_session = dynamic_cast<tensorflow::DirectSession*>(tf_session.get());
	const tensorflow::DeviceMgr* device_mgr = NULL;
	direct_session->LocalDeviceManager(&device_mgr);
	std::vector<tensorflow::Device*> devices = device_mgr->ListDevices();
	for (auto it = devices.begin(); it != devices.end(); ++it) {
		tensorflow::Device* dev = *it;
		const tensorflow::DeviceBase::GpuDeviceInfo* info = dev->tensorflow_gpu_device_info();
		if (info && (info->gpu_id == device_id)) {
			// our nigger.
			dev_ctx = dynamic_cast<tensorflow::GPUDeviceContext*>(info->default_context);
			break;
		}
	}

	assert(dev_ctx);

	perftools::gputools::internal::StreamInterface* si = dev_ctx->device_to_device_stream()->implementation();

	perftools::gputools::cuda::CUDAStream* cuda_stream =
		dynamic_cast<perftools::gputools::cuda::CUDAStream*>(si);

	assert(cuda_stream);

	tf_cuda_ctx = cuda_executor->cuda_context()->context();
	tf_cuda_stream = cuda_stream->cuda_stream();

	assert(tf_cuda_ctx);
	assert(tf_cuda_stream);

	return tensorflow::Status::OK();
}

static CUarray array = NULL;

static void TFVDPAUmap()
{
	CUgraphicsResource resources[1] = { vdpau_cuda_resource };

	CUresult res = cuGraphicsMapResources(1, resources, tf_cuda_stream);
	assert(res == CUDA_SUCCESS);

	res = cuGraphicsSubResourceGetMappedArray(&array, vdpau_cuda_resource, 0, 0);
	assert(res == CUDA_SUCCESS);
}

static void TFVDPAUunmap()
{
	CUgraphicsResource resources[1] = { vdpau_cuda_resource };

	CUresult res = cuGraphicsUnmapResources(1, resources, tf_cuda_stream);
	assert(res == CUDA_SUCCESS);
}

const int net_input_w = OUT_WIDTH;
const int net_input_h = OUT_HEIGHT;
const int net_grid_w = 16;
const int net_grid_h = 9;

struct Obj
{
	Obj() {}
	Obj(int x, int y, float confidence)
	: x(x), y(y), confidence(confidence) {}

	int x;
	int y;
	float confidence;
};

static int tf_mode = 0;
static bool is_dpy = false;

static std::vector<Obj> getTFObjects(const tensorflow::Tensor& objectness_out_tensor, float threshold = 0.5f)
{
	const int nms_x = 1;
	const int nms_y = 1;

	std::vector<Obj> objs;

	const float* objectness_out = objectness_out_tensor.Slice(0, 0).unaligned_flat<float>().data();
	for (int y = 0; y < net_grid_h; ++y) {
		for (int x = 0; x < net_grid_w; ++x) {
			float bg_v = objectness_out[(net_grid_w * 2 * y + x * 2 + 0)];
			float fg_v = objectness_out[(net_grid_w * 2 * y + x * 2 + 1)];
			float confidence = std::exp(fg_v) / (std::exp(fg_v) + std::exp(bg_v));
			if (confidence < threshold)
				continue;
			bool have_better = false;
			for (int oy = y - nms_y; oy <= y + nms_y; ++oy) {
				if (oy < 0)
					continue;
				if (oy >= net_grid_h)
					continue;
				if (have_better)
					break;
				for (int ox = x - nms_x; ox <= x + nms_x; ++ox) {
					if (ox < 0)
						continue;
					if (ox >= net_grid_w)
						continue;
					float test_fg = objectness_out[(net_grid_w * 2 * oy + ox * 2 + 1)];
					if (fg_v < test_fg)
						have_better = true;
				}
			}
			if (!have_better)
				objs.push_back(Obj(x, y, confidence));
		}
	}

	return objs;
}

tensorflow::Tensor is_training_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());

tensorflow::Tensor cuda_inp_tensor;

std::string tensor_name;

tensorflow::Tensor tmp_inp_tensor;

static void feedTFFrame(const cv::Mat& bgra_frame)
{
	tensorflow::Tensor* inp_tensor = NULL;

	if (tf_mode == 2) {
		CUDA_MEMCPY2D cpyinfo;

		cpyinfo.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		cpyinfo.srcHost = 0;
		cpyinfo.srcArray = array;
		cpyinfo.srcXInBytes = 0;
		cpyinfo.srcY = 0;
		cpyinfo.srcPitch = OUT_WIDTH * 4;
		cpyinfo.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyinfo.dstDevice = vdpau_cuda_data;
		cpyinfo.dstXInBytes = 0;
		cpyinfo.dstY = 0;
		cpyinfo.dstPitch = OUT_WIDTH * 4;
		cpyinfo.WidthInBytes = OUT_WIDTH * 4;
		cpyinfo.Height = OUT_HEIGHT;

		CUresult res = cuMemcpy2DAsync(&cpyinfo, tf_cuda_stream);
		assert(res == CUDA_SUCCESS);

		inp_tensor = &cuda_inp_tensor;
	} else {
		tmp_inp_tensor = tensorflow::Tensor(tensorflow::DT_UINT8,
			tensorflow::TensorShape({1, net_input_h, net_input_w, 4}));
		unsigned char* inp_data = tmp_inp_tensor.flat<unsigned char>().data();
		memcpy(inp_data, bgra_frame.data, net_input_w * net_input_h * 4);
		inp_tensor = &tmp_inp_tensor;
	}

	std::vector<tensorflow::Tensor> outputs;

	is_training_tensor.scalar<bool>()() = false;

	auto status = tf_session->Run(
		{{"Placeholder", is_training_tensor}, {tensor_name, *inp_tensor}},
		{"net_objectness_out"}, {}, &outputs);
	assert(status.ok());

	if (is_dpy) {
		auto objs = getTFObjects(outputs[0], 0.5f);

		for (auto it = objs.begin(); it != objs.end(); ++it) {
			printf("x:%d, y:%d, conf:%f\n", it->x, it->y, it->confidence);
		}
	}
}

int main(int argc, char* argv[])
{
	if (argc <= 3) {
		printf("vdapp [file] [cuda] [dpy]\n");
		return 0;
	}

	CompressedVideoStream compressed_vid_stream(argv[1]);
	tf_mode = atoi(argv[2]);
	is_dpy = atoi(argv[3]);

	XEvent event;

	vdp_chroma_type = VDP_CHROMA_TYPE_420;
	vid_width = compressed_vid_stream.get_width();
	vid_height = compressed_vid_stream.get_height();
	colorspace = 1;
	VdpVideoMixerPictureStructure field = VDP_VIDEO_MIXER_PICTURE_STRUCTURE_FRAME;
	procamp.struct_version = VDP_PROCAMP_VERSION;
	procamp.brightness = 0.0;
	procamp.contrast = 1.0;
	procamp.saturation = 1.0;
	procamp.hue = 0.0;
	int status;

	printf("width = %d, height = %d\n", vid_width, vid_height);

	initX();

	status = initVDPAU();
	if (status == -1)
		printf("Error in initializing VdpDevice\n");

	status = createVideoSurface();
	if (status == -1)
		printf("Error in creating VdpVideoSurface\n");

	status = createOutputSurface();
	if (status == -1)
		printf("Error in creating VdpOutputSurface\n");

	status = createVDPMixer(vdp_chroma_type);
	if (status == -1)
		printf("Error in creating VdpVideoMixer\n");

	status = initVDPAUQueue();
	if (status == -1)
		printf("Error in initializing VdpPresentationQueue\n");

	initVDPAUCuda(0);

	tensorflow::Status tf_status = initTF("car.pb", 0);
	if (tf_status != tensorflow::Status::OK()) {
		printf("Error in initializing TF\n");
	}

	createDecoder();

	struct timeval tv1;
	gettimeofday(&tv1, NULL);

	int num_frames = 0;

	cv::Mat tmp_mat;

	if (tf_mode == 2) {
		TFVDPAUmap();

		boost::shared_ptr<VDPAUAllocator> allocator(new VDPAUAllocator((void*)vdpau_cuda_data));
		cuda_inp_tensor = tensorflow::Tensor(allocator.get(), tensorflow::DataType::DT_UINT8, tensorflow::TensorShape({1, net_input_h, net_input_w, 4}));
		tensor_name = "input*0";
	} else {
		tensor_name = "input";
	}

	while (compressed_vid_stream.get_next_frame()) {
		if (is_dpy) {
			vdp_presentation_queue_display(vdp_queue,
				output_surface,
				0, 0,
				0);
		}
		if (tf_mode == 2) {
			feedTFFrame(tmp_mat);
		} else if (tf_mode == 1) {
			feedTFFrame(getBits());
		}
		++num_frames;
	}

	struct timeval tv2, tv_res;
	gettimeofday(&tv2, NULL);

	timersub(&tv2, &tv1, &tv_res);

	double tm = (double)tv_res.tv_sec + (double)tv_res.tv_usec / 1000000.0;

	printf("DONE %f fps!\n", (float)(num_frames / tm));

	if (tf_mode == 2) {
		TFVDPAUunmap();
	}

	cleanupX();

	return 0;
}
