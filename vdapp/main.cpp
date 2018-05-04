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
			300, 300, 5, black, white);
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

	planes[0] = calloc(vid_width * vid_height, sizeof(uint8_t*));
	planes[1] = calloc(vid_width * vid_height / 4, sizeof(uint8_t*));
	planes[2] = calloc(vid_width * vid_height / 4, sizeof(uint8_t*));

	for (uint32_t y = 0; y < vid_height; ++y) {
		for (uint32_t x = 0; x < vid_width; ++x) {
			uint8_t* dy = (uint8_t*)planes[0];
			uint8_t* dcr = (uint8_t*)planes[1];
			uint8_t* dcb = (uint8_t*)planes[2];
			dy[y * vid_height + x] = x + y;
			dcb[(y * vid_height + x) / 4] = 90;
			dcr[(y * vid_height + x) / 4] = 240;
		}
	}

	vdp_st = vdp_video_surface_put_bits_y_cb_cr(video_surface, VDP_YCBCR_FORMAT_YV12,
		planes, pitches);

	free(planes[0]);
	free(planes[1]);
	free(planes[2]);

	if (vdp_st != VDP_STATUS_OK) {
		printf("error = %d\n", (int)vdp_st);
		return -1;
	}

	return 0;
}

static int getBits()
{
	uint32_t **data;
	int i;
	const uint32_t a[1] = {vid_width*4};
	VdpStatus vdp_st;

	data = (uint32_t * * )calloc(1, sizeof(uint32_t *));

	for(i = 0; i < 1; i++)
		data[i] = (uint32_t *)calloc(vid_width*vid_height, sizeof(uint32_t *));

	vdp_st = vdp_output_surface_get_bits_native(output_surface,NULL,
												(void * const*)data,
												 a);

	if (vdp_st != VDP_STATUS_OK)
		return -1;

	return 0;
}

int main(int argc, char* argv[])
{
	XEvent event;

	vdp_chroma_type = VDP_CHROMA_TYPE_420;
	vid_width = 300;
	vid_height = 300;
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

	getBits();
	int i = 500;
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
