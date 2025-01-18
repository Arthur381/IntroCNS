import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

fig, ax = plt.subplots(figsize=(16, 12))

data = np.load("We=0.18, Wi=0.06, rp=8.8ms.npy")

heatmap = ax.imshow(data[0, :,:], cmap='bwr', interpolation='nearest', vmin=-80, vmax=-40)
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

plt.colorbar(heatmap) 
plt.subplots_adjust(bottom=0.2)

def update(frame):
    heatmap.set_array(data[frame, :, :])
    frame_text.set_text(f'Frame: {frame}')
    return heatmap, frame_text

ani = FuncAnimation(fig, update, frames=data.shape[0], interval=20, blit=True)
is_paused = False  # Global variable to track pause state

def pause(event):
    global is_paused
    if is_paused:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    is_paused = not is_paused

ax_pause = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_pause = Button(ax_pause, 'Pause/Resume')
btn_pause.on_clicked(pause)

plt.show()
