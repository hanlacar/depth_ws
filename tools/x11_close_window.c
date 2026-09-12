#include <X11/Xatom.h>
#include <X11/Xlib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "usage: x11_close_window WINDOW_ID\n");
        return 2;
    }
    Display *display = XOpenDisplay(NULL);
    if(!display)
    {
        fprintf(stderr, "cannot open X display\n");
        return 1;
    }
    Window window = (Window)strtoul(argv[1], NULL, 0);
    Atom protocols = XInternAtom(display, "WM_PROTOCOLS", False);
    Atom delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XEvent event;
    memset(&event, 0, sizeof(event));
    event.xclient.type = ClientMessage;
    event.xclient.window = window;
    event.xclient.message_type = protocols;
    event.xclient.format = 32;
    event.xclient.data.l[0] = (long)delete_window;
    event.xclient.data.l[1] = CurrentTime;
    XSendEvent(display, window, False, NoEventMask, &event);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
}
