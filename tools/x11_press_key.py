#!/usr/bin/env python3
"""Send one named key to the currently focused X11 window."""
import ctypes, ctypes.util, os, sys, time
if len(sys.argv)!=2: raise SystemExit("usage: x11_press_key.py KEYNAME")
x11=ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11"));xt=ctypes.cdll.LoadLibrary(ctypes.util.find_library("Xtst"))
x11.XOpenDisplay.restype=ctypes.c_void_p;x11.XStringToKeysym.restype=ctypes.c_ulong
d=x11.XOpenDisplay(os.environ.get("DISPLAY",":1").encode())
if not d: raise SystemExit("cannot open display")
sym=x11.XStringToKeysym(sys.argv[1].encode());code=x11.XKeysymToKeycode(ctypes.c_void_p(d),sym)
if not code: raise SystemExit("unknown key")
xt.XTestFakeKeyEvent(ctypes.c_void_p(d),code,True,0);xt.XTestFakeKeyEvent(ctypes.c_void_p(d),code,False,0)
x11.XFlush(ctypes.c_void_p(d));time.sleep(.2);x11.XCloseDisplay(ctypes.c_void_p(d))
