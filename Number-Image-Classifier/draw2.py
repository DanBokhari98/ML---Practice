from tkinter import *
from tkinter.filedialog import asksaveasfilename as saveAs
import PIL
from PIL import Image, ImageDraw

def save():
    filename=saveAs(title="Save image as...",filetype=(("PNG images","*.png"),("JPEG images","*.jpg"),("GIF images","*.gif")))
    image1.save(filename)

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=5)

    draw.line((lastx, lasty, x, y), fill='black', width=5)
    lastx, lasty = x, y
def clear():
    cv.delete('all')
def exitt():
    exit()

win = Tk()
win.title("Paint - made in Python")
lastx, lasty = None, None

cv = Canvas(win, width=640, height=480, bg='white')
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

save_ = Button(text="Save image", command=save)
save_.pack()

reset=Button(text='Reset canvas',command=clear)
reset.pack(side=LEFT)

_exit=Button(text='Exit',command=exitt)
_exit.pack(side=RIGHT)
win.mainloop()
