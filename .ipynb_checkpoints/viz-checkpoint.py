"""
Pendulum visualizer.

Feed it a list of thetas (the angles you logged in your sim) and it animates the
rod pivoting about the origin. It does NOT know any physics — it just draws each
angle you give it, in order. So whatever your `arm_thetas` actually contains is
exactly what you'll see. If the sim is right, you'll see it swing. If it's wrong,
you'll see the wrong thing (which is useful).

Two things you MUST tell it so it draws the angle the way you meant it:

  units    : 'deg' if your thetas are in degrees, 'rad' if radians.
             (Your notebook stores 45 and calls it degrees -> use 'deg'.)

  zero_ref : where theta = 0 points.
             'down' -> rod hangs straight DOWN at theta=0  (this is the
                       convention your current sim actually uses: its stable
                       equilibrium is at theta=0, i.e. hanging).
             'up'   -> rod points straight UP at theta=0  (the convention in
                       your RWP notes, for later when the wheel goes on).

Positive theta swings counter-clockwise in both cases.

Usage in the notebook
---------------------
    from viz import animate_pendulum

    # writes pendulum.gif and shows it inline:
    animate_pendulum(pendulum.arm_thetas, arm_length=ARM_LENGTH,
                     units='deg', zero_ref='down', dt=DT)

    # custom filename / speed:
    animate_pendulum(pendulum.arm_thetas, arm_length=ARM_LENGTH,
                     units='deg', zero_ref='down', dt=DT,
                     save_path='swing.gif', fps=30)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def _tip_xy(theta_rad, L, zero_ref):
    """Position of the rod tip given the pivot at the origin."""
    if zero_ref == 'down':
        #  theta=0 -> straight down (0, -L);  +theta swings toward +x
        return L * math.sin(theta_rad), -L * math.cos(theta_rad)
    elif zero_ref == 'up':
        #  theta=0 -> straight up (0, +L);   +theta swings toward -x
        return -L * math.sin(theta_rad), L * math.cos(theta_rad)
    raise ValueError("zero_ref must be 'down' or 'up'")


def animate_pendulum(thetas, arm_length=5.0, units='deg', zero_ref='down',
                     dt=0.01, max_frames=400, trail=40, save_path='pendulum.gif',
                     fps=30):
    """
    Animate a pendulum from a list/array of logged angles, save it as a GIF
    (or mp4), and show it inline if you're in a notebook.

    Parameters
    ----------
    thetas      : sequence of angles, one per timestep, in `units`.
    arm_length  : rod length (only sets the drawing scale).
    units       : 'deg' or 'rad' -- how to interpret the numbers in `thetas`.
    zero_ref    : 'down' or 'up'  -- where theta=0 points (see module docstring).
    dt          : sim timestep, seconds. Only used to label the on-screen clock.
    max_frames  : the sim may have 100k steps; we evenly downsample to at most
                  this many frames so the file stays small and watchable.
    trail       : how many past tip positions to draw as a fading trace (0 = off).
    save_path   : output file. '*.gif' -> Pillow, '*.mp4' -> ffmpeg.
    fps         : playback frames/sec of the OUTPUT file. This is playback speed,
                  not sim speed -- the clock overlay shows true sim time, so if
                  the gif looks sped up vs the clock, that's expected and honest.

    Returns
    -------
    The path to the file that was written.
    """
    theta = np.asarray(thetas, dtype=float)
    if units == 'deg':
        theta = np.radians(theta)
    elif units != 'rad':
        raise ValueError("units must be 'deg' or 'rad'")

    n = len(theta)
    if n == 0:
        raise ValueError("thetas is empty -- run the sim first")

    # Evenly downsample to <= max_frames frames.
    step = max(1, n // max_frames)
    idx = np.arange(0, n, step)
    theta_f = theta[idx]
    t_f = idx * dt

    L = arm_length
    tips = np.array([_tip_xy(a, L, zero_ref) for a in theta_f])

    interval_ms = max(1, int(round(1000 / fps)))

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = 1.25 * L
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='0.8', lw=0.8)
    ax.axvline(0, color='0.8', lw=0.8)

    ax.plot(0, 0, 'ko', ms=8, zorder=5)              # pivot
    (rod,) = ax.plot([], [], '-', lw=4, color='#3a6ea5', zorder=4)
    (bob,) = ax.plot([], [], 'o', ms=14, color='#c1440e', zorder=6)
    (trace,) = ax.plot([], [], '-', lw=1, color='#c1440e', alpha=0.35, zorder=3)
    clock = ax.text(0.03, 0.97, '', transform=ax.transAxes, va='top',
                    fontsize=10, family='monospace')

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        trace.set_data([], [])
        clock.set_text('')
        return rod, bob, trace, clock

    def update(i):
        x, y = tips[i]
        rod.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        if trail:
            lo = max(0, i - trail)
            trace.set_data(tips[lo:i + 1, 0], tips[lo:i + 1, 1])
        deg_here = math.degrees(theta_f[i])
        clock.set_text(f"t = {t_f[i]:7.2f} s\n"
                       f"theta = {deg_here:7.2f} deg\n"
                       f"({zero_ref}=0)")
        return rod, bob, trace, clock

    anim = FuncAnimation(fig, update, frames=len(theta_f), init_func=init,
                         interval=interval_ms, blit=True)

    if save_path.endswith('.gif'):
        anim.save(save_path, writer=PillowWriter(fps=fps))
    elif save_path.endswith('.mp4'):
        anim.save(save_path, writer=FFMpegWriter(fps=fps))
    else:
        raise ValueError("save_path must end in .gif or .mp4")
    plt.close(fig)
    print(f"saved -> {save_path}  "
          f"({len(theta_f)} frames @ {fps} fps = {len(theta_f)/fps:.1f}s of playback, "
          f"covering {t_f[-1]:.0f}s of sim time)")

    # If we're in a notebook, drop it inline too.
    try:
        from IPython.display import Image, Video, display
        display(Image(filename=save_path) if save_path.endswith('.gif')
                else Video(save_path, embed=True))
    except Exception:
        pass
    return save_path
