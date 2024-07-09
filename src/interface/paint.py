from tkinter import *
from PIL import ImageTk, Image, ImageGrab
import os
import sys

# Añade el directorio raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.canvas_processing import CanvasProcessing

class Paint:
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.setup_gui()
        self.setup_canvas()
        self.setup_tools()
        self.setup_events()

    def setup_gui(self):
        self.root = Tk()
        self.root.title('Paint')
        self.root.geometry('1200x450')  # Cambiar a 1200x450
        self.root.maxsize(1200, 450)
        self.root.minsize(1200, 450)
        self.root.configure(bg='#121212')  # Modo oscuro

    def setup_canvas(self):
        self.c = Canvas(self.root, bg='#FFFFFF', width=1100, height=450, relief=RIDGE, borderwidth=0)  # Ajustar dimensiones del canvas
        self.c.pack(side=RIGHT)

    def setup_tools(self):
        self.paint_tools = Frame(self.root, width=100, height=450, relief=RIDGE, borderwidth=2, bg='#121212')  # Ajustar dimensiones del frame
        self.paint_tools.pack(side=LEFT, fill=Y)

        self.pen_logo = ImageTk.PhotoImage(Image.open('src/interface/assets/pen.png').resize((36, 36)))
        self.pen_button = Button(self.paint_tools, image=self.pen_logo, command=self.use_pen, relief=FLAT)
        self.pen_button.pack(pady=10)

        self.eraser_logo = ImageTk.PhotoImage(Image.open('src/interface/assets/eraser.png').resize((36, 36)))
        self.eraser_button = Button(self.paint_tools, image=self.eraser_logo, command=self.use_eraser, relief=FLAT)
        self.eraser_button.pack(pady=10)

        self.save_button = Button(self.paint_tools, text="Reconocer", command=self.recognize_characters, bg='#333', fg='white', relief=FLAT)
        self.save_button.pack(pady=20)

        self.text_entry = Entry(self.paint_tools, width=30, font=('verdana', 10), bg='#212121', fg='white', insertbackground='white')
        self.text_entry.pack(pady=20)

    def setup_events(self):
        self.old_x, self.old_y = None, None
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
        self.c.delete('all')
        self.text_entry.delete(0, END)
        self.use_pen()

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=FLAT)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                width=10, fill=paint_color,
                                capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def recognize_characters(self):
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        image_path = 'temp.png'
        image.save(image_path)

        recognizer = CanvasProcessing('src/models/saved_models/model.keras')
        recognized_text = recognizer.process_image(image_path)

        print(f"Texto reconocido: {recognized_text}")

        self.text_entry.delete(0, END)
        self.text_entry.insert(0, recognized_text)

    def run(self):
        self.root.mainloop()
