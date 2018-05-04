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

static Display* dpy = NULL;
static Window win = 0;
static GC gc = 0;

void initX()
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

int main(int argc, char* argv[])
{
	return 0;
}
