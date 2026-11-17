import sys, os, win32com.client, pythoncom
src = os.path.abspath(sys.argv[1]); out = os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True)
only = set(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
pythoncom.CoInitialize()
app = win32com.client.DispatchEx("PowerPoint.Application")
pres = app.Presentations.Open(src, True, False, False)   # read-only, untitled no, no window
for i in range(1, pres.Slides.Count + 1):
    if only and i not in only: continue
    pres.Slides(i).Export(os.path.join(out, f"slide-{i:02d}.png"), "PNG", 1600, 900)
print("slides:", pres.Slides.Count)
pres.Close(); app.Quit()
