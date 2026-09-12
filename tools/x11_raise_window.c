#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: x11_raise_window WINDOW_ID\n");
    return 2;
  }
  Display *display = XOpenDisplay(NULL);
  if (!display) return 3;
  Window window = (Window)strtoul(argv[1], NULL, 0);
  Atom active = XInternAtom(display, "_NET_ACTIVE_WINDOW", False);
  XEvent event = {0};
  event.xclient.type = ClientMessage;
  event.xclient.message_type = active;
  event.xclient.display = display;
  event.xclient.window = window;
  event.xclient.format = 32;
  event.xclient.data.l[0] = 1;
  event.xclient.data.l[1] = CurrentTime;
  XSendEvent(display, DefaultRootWindow(display), False,
             SubstructureRedirectMask | SubstructureNotifyMask, &event);
  XMapRaised(display, window);
  XSetInputFocus(display, window, RevertToParent, CurrentTime);
  XFlush(display);
  XCloseDisplay(display);
  return 0;
}
