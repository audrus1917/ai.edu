import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 1. Setup Tkinter window
root = tk.Tk()
root.title("Matplotlib in Tkinter")

# 2. Create Matplotlib Figure and Plot
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Simple Plot")

# 3. Embed Figure into Tkinter Canvas
canvas = FigureCanvasTkAgg(fig, master=root)  
canvas.draw()

# 4. Place Canvas in the window
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()