#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

extern int XTestFakeKeyEvent(Display *, unsigned int, Bool, unsigned long);

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: x11_send_keys WINDOW_ID KEYSYM...\n");
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
  XSendEvent(display, DefaultRootWindow(display), False,
             SubstructureRedirectMask | SubstructureNotifyMask, &event);
  XFlush(display);
  usleep(250000);
  for (int i=2; i<argc; ++i) {
    const char *name = argv[i];
    Bool alt = strncmp(name, "Alt+", 4) == 0;
    if (alt) name += 4;
    KeySym symbol = XStringToKeysym(name);
    if (symbol == NoSymbol) { fprintf(stderr, "unknown keysym %s\n", argv[i]); continue; }
    KeyCode code = XKeysymToKeycode(display, symbol);
    KeyCode altCode = XKeysymToKeycode(display, XK_Alt_L);
    if (alt) XTestFakeKeyEvent(display, altCode, True, CurrentTime);
    XTestFakeKeyEvent(display, code, True, CurrentTime);
    XTestFakeKeyEvent(display, code, False, CurrentTime);
    if (alt) XTestFakeKeyEvent(display, altCode, False, CurrentTime);
    XFlush(display);
    usleep(180000);
  }
  XCloseDisplay(display);
  return 0;
}
