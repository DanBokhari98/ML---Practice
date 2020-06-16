# DRAW IMAGES FIRST #
from tkinter import *

canvas_width = 28
canvas_height = 28

def paint( event ):
   python_green = "#476042"
   x1, y1 = ( event.x - 1 ), ( event.y - 1 )
   x2, y2 = ( event.x + 1 ), ( event.y + 1 )
   w.create_oval( x1, y1, x2, y2, fill = python_green )

master = Tk()
master.title( "Painting using Ovals" )
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )
print("HEY IM HERE")
message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )


master.mainloop()

w.update()
w.postscript(file="img.png", colormode='color')

master.mainloop()

# END OF DRAWING AN IMAGE #
