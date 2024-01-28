from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont

from tkinter_display import create_display_window
pixels_size = (128, 64)
try:
    import board
    import adafruit_ssd1306
    i2c = board.I2C()
    oled = adafruit_ssd1306.SSD1306_I2C(pixels_size[0], pixels_size[1], i2c)
except ImportError:
    oled = None


char_h = 11
rpi_font_poath = "fonts/DejaVuSans.ttf"
font = ImageFont.truetype(rpi_font_poath, char_h)
max_x, max_y = 22, 5
display_lines = [""]

def _display_update():
    """ Show lines on the screen """
    global oled
    image = Image.new("1", pixels_size)
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(display_lines):
        draw.text((0, y*char_h), line, font=font, fill=255, align="left")

    if oled:
        oled.fill(0)
        oled.image(image)
        oled.show()


def add_display_line(text: str):
    """ Add new line with scrolling """
    global display_lines
    # Split line to chunks according to screen width
    text_chunks = [text[i: i+max_x] for i in range(0, len(text), max_x)]
    for text in text_chunks:
        for line in text.split("\n"):
            display_lines.append(line)
            display_lines = display_lines[-max_y:]
    _display_update()

def add_display_tokens(text: str):
    """ Add new tokens with or without extra line break """
    global display_lines
    last_line = display_lines.pop()
    new_line = last_line + text
    add_display_line(new_line)

def schedule_updates(index=0):
    if index < 20:
        add_display_line(f"{datetime.now().strftime('%H:%M:%S')}: Line-{index}")
        update_display(display_lines)
        window.after(200, lambda: schedule_updates(index + 1))

if oled:
    for p in range(20):
        add_display_line(f"{datetime.now().strftime('%H:%M:%S')}: Line-{p}")
        time.sleep(0.2)
else:
    window, update_display = create_display_window()
    schedule_updates()  # Schedule the first update
    window.mainloop()
