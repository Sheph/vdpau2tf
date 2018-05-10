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

#include <cuda.h>
#include <cudaVDPAU.h>

#undef Status

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

#define MY_VID_WIDTH 128
#define MY_VID_HEIGHT 72

static Display* dpy = NULL;
static Window win = 0;
static GC gc = 0;

static VdpOutputSurface output_surface;
static VdpVideoSurface video_surface;
static VdpProcamp procamp;
static VdpVideoMixer video_mixer;
static uint32_t vid_width, vid_height;
static VdpChromaType vdp_chroma_type;
static VdpDevice vdp_device;
static int colorspace;
static VdpPresentationQueueTarget vdp_target;
static VdpPresentationQueue vdp_queue;

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
static VdpGenerateCSCMatrix *vdp_generate_csc_matrix;

static void initX()
{
	unsigned long black, white;

	dpy = XOpenDisplay((char*)0);
	black = BlackPixel(dpy, DefaultScreen(dpy)),
	white = WhitePixel(dpy, DefaultScreen(dpy));
	win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0,
			MY_VID_WIDTH, MY_VID_HEIGHT, 5, black, white);
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

	vdp_st = vdp_video_surface_create(vdp_device, vdp_chroma_type,
										   vid_width, vid_height,
										   &video_surface);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

static int createOutputSurface()
{
	VdpStatus vdp_st;

	vdp_st = vdp_output_surface_create(vdp_device, VDP_RGBA_FORMAT_B8G8R8A8,
										   vid_width, vid_height,
										   &output_surface);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

static int putBits()
{
	VdpStatus vdp_st;

	void* planes[3];
	uint32_t pitches[3] = {vid_width, vid_width / 2, vid_width / 2};

	cv::Mat frame = cv::imread("car.png");
	cv::resize(frame, frame, cv::Size(vid_width, vid_height), 0, 0, cv::INTER_AREA);
	cv::cvtColor(frame, frame, CV_BGR2YUV_YV12);

	planes[0] = frame.data;
	planes[1] = frame.data + vid_width * vid_height;
	planes[2] = frame.data + vid_width * vid_height + vid_width * vid_height / 4;

	vdp_st = vdp_video_surface_put_bits_y_cb_cr(video_surface, VDP_YCBCR_FORMAT_YV12,
		planes, pitches);

	if (vdp_st != VDP_STATUS_OK) {
		printf("error = %d\n", (int)vdp_st);
		return -1;
	}

	return 0;
}

static cv::Mat getBits()
{
	VdpStatus vdp_st;

	void* planes[1];
	uint32_t pitches[1] = {vid_width * 4};

	cv::Mat data(vid_height, vid_width, CV_8UC4);

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

	res = cuMemAlloc(&vdpau_cuda_data, vid_width * vid_height * 4);
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

static void* TFVDPAUmap()
{
	CUgraphicsResource resources[1] = { vdpau_cuda_resource };

	CUresult res = cuGraphicsMapResources(1, resources, tf_cuda_stream);
	assert(res == CUDA_SUCCESS);

	CUarray array = NULL;

	res = cuGraphicsSubResourceGetMappedArray(&array, vdpau_cuda_resource, 0, 0);
	assert(res == CUDA_SUCCESS);

	CUDA_MEMCPY2D cpyinfo;

	cpyinfo.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cpyinfo.srcHost = 0;
	cpyinfo.srcArray = array;
	cpyinfo.srcXInBytes = 0;
	cpyinfo.srcY = 0;
	cpyinfo.srcPitch = vid_width * 4;
	cpyinfo.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	cpyinfo.dstDevice = vdpau_cuda_data;
	cpyinfo.dstXInBytes = 0;
	cpyinfo.dstY = 0;
	cpyinfo.dstPitch = vid_width * 4;
	cpyinfo.WidthInBytes = vid_width * 4;
	cpyinfo.Height = vid_height;

	res = cuMemcpy2DAsync(&cpyinfo, tf_cuda_stream);
	assert(res == CUDA_SUCCESS);

	return (void*)vdpau_cuda_data;
}

static void TFVDPAUunmap()
{
	CUgraphicsResource resources[1] = { vdpau_cuda_resource };

	CUresult res = cuGraphicsUnmapResources(1, resources, tf_cuda_stream);
	assert(res == CUDA_SUCCESS);
}

const int net_input_w = 128;
const int net_input_h = 72;
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

static void feedTFFrame(const cv::Mat& bgra_frame)
{
	void* vdpau_ptr = TFVDPAUmap();

	boost::shared_ptr<VDPAUAllocator> allocator(new VDPAUAllocator(vdpau_ptr));
	tensorflow::Tensor inp_tensor(allocator.get(), tensorflow::DataType::DT_UINT8, tensorflow::TensorShape({1, net_input_h, net_input_w, 4}));

	/*tensorflow::Tensor inp_tensor(tensorflow::DT_UINT8,
		tensorflow::TensorShape({1, net_input_h, net_input_w, 4}));
	unsigned char* inp_data = inp_tensor.flat<unsigned char>().data();
	memcpy(inp_data, bgra_frame.data, net_input_w * net_input_h * 4);*/

	std::vector<tensorflow::Tensor> outputs;

	tensorflow::Tensor is_training_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
	is_training_tensor.scalar<bool>()() = false;

	auto status = tf_session->Run(
		{{"Placeholder", is_training_tensor}, {"input*0", inp_tensor}},
		{"net_objectness_out", "net_convlast_copy"}, {}, &outputs);
	assert(status.ok());

	auto objs = getTFObjects(outputs[0], 0.5f);

	for (auto it = objs.begin(); it != objs.end(); ++it) {
		printf("x:%d, y:%d, conf:%f\n", it->x, it->y, it->confidence);
	}

	TFVDPAUunmap();
}

int main(int argc, char* argv[])
{
	XEvent event;

	vdp_chroma_type = VDP_CHROMA_TYPE_420;
	vid_width = MY_VID_WIDTH;
	vid_height = MY_VID_HEIGHT;
	colorspace = 1;
	VdpVideoMixerPictureStructure field = VDP_VIDEO_MIXER_PICTURE_STRUCTURE_FRAME;
	procamp.struct_version = VDP_PROCAMP_VERSION;
	procamp.brightness = 0.0;
	procamp.contrast = 1.0;
	procamp.saturation = 1.0;
	procamp.hue = 0.0;
	int status;

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

	status = putBits();
	if (status == -1)
		printf("Error in Putting data on VdpVideoSurface\n");

	vdp_video_mixer_render(video_mixer, VDP_INVALID_HANDLE, 0,
										field, 0, (VdpVideoSurface*)VDP_INVALID_HANDLE,
										video_surface,
										0, (VdpVideoSurface*)VDP_INVALID_HANDLE,
										NULL,
										output_surface,
										NULL, NULL, 0, NULL);

	cv::Mat frame = getBits();
	feedTFFrame(frame);
	feedTFFrame(frame);
	feedTFFrame(frame);

	int i = 1;
	while (i) {
	   vdp_presentation_queue_display(vdp_queue,
											  output_surface,
											  0, 0,
											  0);
	   i--;
	}

	while (true) {
	   XNextEvent(dpy, &event);
	   if (event.type == KeyPress) {
		   cleanupX();
		   break;
	   }
	}

	return 0;
}
