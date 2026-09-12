#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern int XTestFakeMotionEvent(Display *, int, int, int, unsigned long);
extern int XTestFakeButtonEvent(Display *, unsigned int, Bool, unsigned long);

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  Display *display=XOpenDisplay(NULL);
  if (!display) return 3;
  XTestFakeMotionEvent(display, DefaultScreen(display), atoi(argv[1]), atoi(argv[2]), CurrentTime);
  XFlush(display); usleep(120000);
  XTestFakeButtonEvent(display, 1, True, CurrentTime);
  XTestFakeButtonEvent(display, 1, False, CurrentTime);
  XFlush(display); usleep(200000);
  XCloseDisplay(display);
  return 0;
}
