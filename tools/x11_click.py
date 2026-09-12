#!/usr/bin/env python3
"""Click one absolute coordinate on the active X11 desktop."""
import ctypes,ctypes.util,os,sys,time
if len(sys.argv)!=3:raise SystemExit('usage: x11_click.py X Y')
x11=ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11'));xt=ctypes.cdll.LoadLibrary(ctypes.util.find_library('Xtst'));x11.XOpenDisplay.restype=ctypes.c_void_p
d=x11.XOpenDisplay(os.environ.get('DISPLAY',':1').encode())
if not d:raise SystemExit('cannot open display')
xt.XTestFakeMotionEvent(ctypes.c_void_p(d),-1,int(sys.argv[1]),int(sys.argv[2]),0)
xt.XTestFakeButtonEvent(ctypes.c_void_p(d),1,True,0);xt.XTestFakeButtonEvent(ctypes.c_void_p(d),1,False,0)
x11.XFlush(ctypes.c_void_p(d));time.sleep(.2);x11.XCloseDisplay(ctypes.c_void_p(d))
