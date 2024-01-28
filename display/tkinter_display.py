import tkinter as tk
from tkinter import font

# Constants for your simulated display
DISPLAY_WIDTH = 128
DISPLAY_HEIGHT = 64
FONT_SIZE = 10  # Adjust as needed
FONT_NAME = "DejaVu Sans Mono"  # Use a monospace font for better simulation

def create_display_window():
    window = tk.Tk()
    window.title("OLED Display Simulation")

    # Create a canvas to draw text
    canvas = tk.Canvas(window, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg="black")
    canvas.pack()

    # Load the font
    display_font = font.Font(family=FONT_NAME, size=FONT_SIZE)

    # Function to update the display
    def update_display(text_lines):
        canvas.delete("all")  # Clear the canvas
        y = 0
        for line in text_lines:
            canvas.create_text(2, y, anchor="nw", text=line, fill="white", font=display_font)
            y += FONT_SIZE

    return window, update_display

if __name__=="__main__":
    # Create the window and the update function
    window, update_display = create_display_window()

    # Example usage
    test_lines = ["Line 1", "Line 2", "Line 3"]
    update_display(test_lines)

    # Run the Tkinter loop
    window.mainloop()