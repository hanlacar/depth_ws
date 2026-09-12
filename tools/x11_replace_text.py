#!/usr/bin/env python3
"""Replace text in the currently focused X11 input (ASCII only)."""
import ctypes,ctypes.util,os,sys,time
if len(sys.argv)!=2:raise SystemExit('usage: x11_replace_text.py TEXT')
x11=ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11'));xt=ctypes.cdll.LoadLibrary(ctypes.util.find_library('Xtst'));x11.XOpenDisplay.restype=ctypes.c_void_p;x11.XStringToKeysym.restype=ctypes.c_ulong
d=x11.XOpenDisplay(os.environ.get('DISPLAY',':1').encode())
def code(name):return x11.XKeysymToKeycode(ctypes.c_void_p(d),x11.XStringToKeysym(name.encode()))
def key(k,down):xt.XTestFakeKeyEvent(ctypes.c_void_p(d),k,down,0)
ctrl=code('Control_L');key(ctrl,True);key(code('a'),True);key(code('a'),False);key(ctrl,False)
for ch in sys.argv[1]:
 name={'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','.':'period','-':'minus'}[ch]
 k=code(name);key(k,True);key(k,False)
x11.XFlush(ctypes.c_void_p(d));time.sleep(.2);x11.XCloseDisplay(ctypes.c_void_p(d))
