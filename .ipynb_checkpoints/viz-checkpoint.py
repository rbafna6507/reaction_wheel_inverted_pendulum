"""
Pendulum visualizer.

Feed it a list of thetas (the angles you logged in your sim) and it animates the
rod pivoting about the origin, saves a GIF, and shows it inline in a notebook.
It does NOT know any physics -- it just draws the angles you give it, in order.
So whatever your `arm_thetas` actually contains is exactly what you'll see.

Works with BOTH systems:
  * plain arm  -> pass just the arm angles.
  * reaction wheel -> also pass the wheel angles; it draws the wheel at the arm
    tip with a spinning spoke so you can watch it turn (and watch it whir faster
    as it saturates).

Two things you MUST tell it so it draws the angle the way you meant it (same as
before):

  units    : 'deg' if your thetas are in degrees, 'rad' if radians.
  zero_ref : 'down' -> rod hangs straight DOWN at theta=0   (your sim's stable
                       rest; balancing at pi means the rod ends up pointing UP)
             'up'   -> rod points straight UP at theta=0

Easiest calls
-------------
    from viz import animate_system

    animate_system(pendulum, units='deg', zero_ref='down', dt=DT)   # plain arm
    animate_system(rwp,       units='rad', zero_ref='down', dt=DT)   # reaction wheel

`animate_system` reads arm_thetas / arm_length (and wheel_thetas / wheel_radius
if the object has them) straight off your object, so the same call handles both.

Or call the underlying function directly:
    from viz import animate_pendulum
    animate_pendulum(rwp.arm_thetas, arm_length=ARM_LENGTH, units='rad',
                     zero_ref='down', dt=DT,
                     wheel_thetas=rwp.wheel_thetas, wheel_radius=WHEEL_RADIUS)
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
                     fps=30, wheel_thetas=None, wheel_radius=None):
    """
    Animate a pendulum from a list/array of logged angles, save it as a GIF
    (or mp4), and show it inline if you're in a notebook.

    Parameters
    ----------
    thetas       : arm angles, one per timestep, in `units`.
    arm_length   : rod length (sets the drawing scale; wheel is drawn at the tip).
    units        : 'deg' or 'rad' -- how to read the numbers in `thetas`
                   (and in `wheel_thetas`, if given).
    zero_ref     : 'down' or 'up' -- where theta=0 points (see module docstring).
    dt           : sim timestep, seconds. Labels the on-screen clock.
    max_frames   : downsample to at most this many frames so the file stays small.
    trail        : how many past tip positions to draw as a fading trace (0=off).
    save_path    : output file. '*.gif' -> Pillow, '*.mp4' -> ffmpeg.
    fps          : playback frames/sec of the OUTPUT file (playback speed, not
                   sim speed -- the clock shows true sim time).
    wheel_thetas : OPTIONAL wheel spin angles, one per timestep. If given, a wheel
                   is drawn at the arm tip with a rotating spoke. Same length /
                   units as `thetas` (extra samples are trimmed to match).
    wheel_radius : wheel drawing radius (physical units). Defaults to 0.25*arm
                   length if a wheel is drawn but no radius is supplied.

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

    have_wheel = wheel_thetas is not None and len(wheel_thetas) > 0
    if have_wheel:
        wtheta = np.asarray(wheel_thetas, dtype=float)
        if units == 'deg':
            wtheta = np.radians(wtheta)
        m = min(n, len(wtheta))          # keep arm and wheel in lockstep
        theta, wtheta, n = theta[:m], wtheta[:m], m
        if wheel_radius is None:
            wheel_radius = 0.25 * arm_length

    # Evenly downsample to <= max_frames frames.
    step = max(1, n // max_frames)
    idx = np.arange(0, n, step)
    theta_f = theta[idx]
    t_f = idx * dt
    L = arm_length
    tips = np.array([_tip_xy(a, L, zero_ref) for a in theta_f])
    if have_wheel:
        wtheta_f = wtheta[idx]

    interval_ms = max(1, int(round(1000 / fps)))

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = 1.25 * L + (wheel_radius if have_wheel else 0.0)
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
    artists = [rod, bob, trace, clock]

    if have_wheel:
        circ = np.linspace(0, 2 * np.pi, 60)         # unit circle for the rim
        (rim,) = ax.plot([], [], '-', lw=2, color='#2a2a2a', zorder=7)
        (spoke,) = ax.plot([], [], '-', lw=2.5, color='#e0a030', zorder=8)
        (hub,) = ax.plot([], [], 'o', ms=6, color='#e0a030', zorder=9)
        artists += [rim, spoke, hub]

    def init():
        for a in artists:
            if hasattr(a, 'set_data'):
                a.set_data([], [])
        clock.set_text('')
        return artists

    def update(i):
        x, y = tips[i]
        rod.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        if trail:
            lo = max(0, i - trail)
            trace.set_data(tips[lo:i + 1, 0], tips[lo:i + 1, 1])
        txt = (f"t = {t_f[i]:7.2f} s\n"
               f"theta = {math.degrees(theta_f[i]):7.2f} deg\n"
               f"({zero_ref}=0)")
        if have_wheel:
            r = wheel_radius
            rim.set_data(x + r * np.cos(circ), y + r * np.sin(circ))
            phi = wtheta_f[i]
            spoke.set_data([x, x + r * math.cos(phi)], [y, y + r * math.sin(phi)])
            hub.set_data([x + r * math.cos(phi)], [y + r * math.sin(phi)])
            txt += f"\nwheel = {math.degrees(phi):9.0f} deg"
        clock.set_text(txt)
        return artists

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


def animate_system(system, **kwargs):
    """
    Animate a System or ReactionWheelSystem object directly -- works for both.

    Pulls arm_thetas / arm_length off the object, and if the object also carries
    wheel_thetas / wheel_radius (i.e. it's a reaction wheel), draws the wheel too.
    Pass units / zero_ref / dt / save_path / fps as keyword args.

        animate_system(pendulum, units='deg', zero_ref='down', dt=DT)
        animate_system(rwp,       units='rad', zero_ref='down', dt=DT)
    """
    wheel_thetas = getattr(system, 'wheel_thetas', None)
    if not wheel_thetas:                 # None or empty list -> plain arm
        wheel_thetas = None
    return animate_pendulum(
        system.arm_thetas,
        arm_length=getattr(system, 'arm_length', 5.0),
        wheel_thetas=wheel_thetas,
        wheel_radius=getattr(system, 'wheel_radius', None),
        **kwargs,
    )
