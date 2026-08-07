"""
Pendulum visualizer.

Feed it a list of thetas (the angles you logged in your sim); it animates the
rod pivoting about the origin, saves an MP4, and shows it inline in a notebook.
It does NOT know any physics -- it draws the angles you give it, in order.

Works with BOTH systems:
  * plain arm     -> pass just the arm angles.
  * reaction wheel -> also pass the wheel angles; it draws the wheel at the arm
    tip with a spinning spoke (watch it whir faster as it saturates).

--------------------------------------------------------------------------------
SMOOTHNESS -- read this once, it explains every knob
--------------------------------------------------------------------------------
Choppiness is NOT about your sim's dt. It's about how much SIM TIME passes
between two drawn frames:

    sim_seconds_per_frame = (window of sim time shown) / (number of frames drawn)

The old default drew 400 frames across all 1000 s of sim = 2.5 s/frame, so the
rod jumped ~50 deg per frame -- that's aliasing, not motion. The governing knobs:

    speed : how many SIM seconds play per ONE real second of video.
            speed=1  -> real time (smoothest).  speed=50 -> 50x fast-forward.
            sim_seconds_per_frame = speed / fps, so LOW speed = smooth.
    fps   : output frame rate (60 is plenty).
    window: which slice of sim time to show (t_start .. t_end), or full=True.

The hard tradeoff (unavoidable): you canNOT show all 1000 s AND be smooth AND
keep the clip short. Real-time-smooth full run = 1000 s * 60 fps = 60k frames =
a ~17-minute video. So either watch a short WINDOW at real time (default), or
watch the FULL run as a sped-up time-lapse (full=True) and accept some chop.

--------------------------------------------------------------------------------
Easiest calls
--------------------------------------------------------------------------------
    from viz import animate_system

    animate_system(pendulum, units='deg', zero_ref='down', dt=DT)   # plain arm
    animate_system(rwp,       units='rad', zero_ref='down', dt=DT)   # reaction wheel

By default you get a smooth, real-time MP4 of the first ~20 s. To see the whole
run (time-lapse):  animate_system(rwp, ..., full=True)

`units` : 'deg' or 'rad' -- how to read your numbers.
`zero_ref` : 'down' (theta=0 hangs down; balancing at pi means the rod ends up
             pointing UP) or 'up' (theta=0 points up).
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
                     dt=0.01, save_path='pendulum.mp4',
                     fps=60, speed=1.0, t_start=0.0, t_end=None, full=False,
                     max_frames=6000, dpi=150, crf=18, trail=60,
                     wheel_thetas=None, wheel_speeds=None, wheel_radius=None):
    """
    Animate logged pendulum angles -> MP4 (or GIF), shown inline in a notebook.

    Timing (see the module docstring for the why):
    speed        : sim-seconds per real playback-second. 1.0 = real time (smooth).
                   Raised automatically if the window needs more than max_frames.
    fps          : output frame rate.
    t_start,t_end: sim-time window to show, in seconds. t_end=None -> a ~20 s
                   real-time slice starting at t_start (smooth by default).
    full         : True -> show the ENTIRE run as a time-lapse (overrides t_end;
                   speed is auto-picked to fit max_frames).
    max_frames   : hard cap on frames drawn (keeps files/renders sane). If a
                   window needs more, speed is bumped up (=> a time-lapse).

    Quality:
    dpi          : output resolution = 5in * dpi (150 -> 750px). 200 is crisper.
    crf          : H.264 quality for MP4, lower = better (18 is ~visually lossless
                   here; 23 default). Ignored for GIF.
    trail        : number of past tip positions drawn as a fading trace (0=off).

    Geometry:
    arm_length   : rod length / drawing scale.
    units        : 'deg' or 'rad'.
    zero_ref     : 'down' or 'up'.
    wheel_thetas : optional wheel spin angles (same length/units as thetas) -> a
                   wheel with a spinning spoke is drawn at the arm tip.
    wheel_speeds : optional wheel angular velocities (rad/s, or deg/s if units=
                   'deg'). Shown in the readout so you can watch it saturate. If
                   omitted, it's derived by differentiating wheel_thetas.
    wheel_radius : wheel drawing radius; defaults to 0.25*arm_length if omitted.

    Returns the path to the file written.
    """
    theta = np.asarray(thetas, dtype=float)
    if units == 'deg':
        theta = np.radians(theta)
    elif units != 'rad':
        raise ValueError("units must be 'deg' or 'rad'")
    n = len(theta)
    if n < 2:
        raise ValueError("need at least 2 thetas -- run the sim first")

    have_wheel = wheel_thetas is not None and len(wheel_thetas) > 0
    if have_wheel:
        wtheta = np.asarray(wheel_thetas, dtype=float)
        if units == 'deg':
            wtheta = np.radians(wtheta)
        # wheel SPEED for the readout: use logged speeds if given, else
        # differentiate the angle (rad/s). This is what shows saturation.
        if wheel_speeds is not None and len(wheel_speeds) > 0:
            wspeed = np.asarray(wheel_speeds, dtype=float)
            if units == 'deg':
                wspeed = np.radians(wspeed)
        else:
            wspeed = np.gradient(wtheta, dt)
        m = min(n, len(wtheta), len(wspeed))
        theta, wtheta, wspeed, n = theta[:m], wtheta[:m], wspeed[:m], m
        if wheel_radius is None:
            wheel_radius = 0.25 * arm_length

    # ---- decide which slice of sim time to show, and how many frames ----------
    total = (n - 1) * dt
    if full:
        a, b = 0.0, total
    elif t_end is None:
        a = max(0.0, t_start)
        b = min(total, a + 20.0)              # default: ~20 s, real time
    else:
        a = max(0.0, t_start)
        b = min(total, t_end)
    if b <= a:
        a, b = 0.0, total                     # fall back to the whole run
    window = b - a

    frames = max(2, int(round(window * fps / speed)))
    time_lapsed = False
    if frames > max_frames:                   # too many -> fast-forward to fit
        frames = max_frames
        speed = window * fps / frames
        time_lapsed = True
    sspf = window / frames                    # sim seconds per frame (= speed/fps)

    sim_t = np.linspace(a, b, frames)
    idx = np.clip(np.round(sim_t / dt).astype(int), 0, n - 1)   # np.round, not int()
    theta_f = theta[idx]
    L = arm_length
    tips = np.array([_tip_xy(t, L, zero_ref) for t in theta_f])
    if have_wheel:
        wtheta_f = wtheta[idx]
        wspeed_f = wspeed[idx]

    # ---- keep output dimensions even (h264 yuv420p needs it) ------------------
    figsize = 5.0
    if int(round(figsize * dpi)) % 2:
        dpi += 1

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    lim = 1.25 * L + (wheel_radius if have_wheel else 0.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='0.8', lw=0.8)
    ax.axvline(0, color='0.8', lw=0.8)

    ax.plot(0, 0, 'ko', ms=8, zorder=5)                         # pivot
    (rod,) = ax.plot([], [], '-', lw=4, color='#3a6ea5', zorder=4)
    (bob,) = ax.plot([], [], 'o', ms=14, color='#c1440e', zorder=6)
    (trace,) = ax.plot([], [], '-', lw=1.2, color='#c1440e', alpha=0.35, zorder=3)
    clock = ax.text(0.03, 0.97, '', transform=ax.transAxes, va='top',
                    fontsize=10, family='monospace', zorder=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    artists = [rod, bob, trace, clock]

    if have_wheel:
        circ = np.linspace(0, 2 * np.pi, 60)
        (rim,) = ax.plot([], [], '-', lw=2, color='#2a2a2a', zorder=7)
        (spoke,) = ax.plot([], [], '-', lw=2.5, color='#e0a030', zorder=8)
        (hub,) = ax.plot([], [], 'o', ms=6, color='#e0a030', zorder=9)
        artists += [rim, spoke, hub]

    def init():
        for art in artists:
            if hasattr(art, 'set_data'):
                art.set_data([], [])
        clock.set_text('')
        return artists

    def update(i):
        x, y = tips[i]
        rod.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        if trail:
            lo = max(0, i - trail)
            trace.set_data(tips[lo:i + 1, 0], tips[lo:i + 1, 1])
        txt = (f"t = {sim_t[i]:7.2f} s   ({speed:.0f}x)\n"
               f"theta = {math.degrees(theta_f[i]):7.2f} deg\n"
               f"({zero_ref}=0)")
        if have_wheel:
            r = wheel_radius
            rim.set_data(x + r * np.cos(circ), y + r * np.sin(circ))
            phi = wtheta_f[i]
            spoke.set_data([x, x + r * math.cos(phi)], [y, y + r * math.sin(phi)])
            hub.set_data([x + r * math.cos(phi)], [y + r * math.sin(phi)])
            w = wspeed_f[i]
            txt += f"\nwheel = {w:7.1f} rad/s ({w/(2*math.pi):+.1f} rev/s)"
        clock.set_text(txt)
        return artists

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=1000 / fps, blit=True)

    if save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, codec='h264',
                              extra_args=['-crf', str(crf), '-pix_fmt', 'yuv420p'])
        anim.save(save_path, writer=writer, dpi=dpi)
    elif save_path.endswith('.gif'):
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        raise ValueError("save_path must end in .mp4 or .gif")
    plt.close(fig)

    playback = frames / fps
    print(f"saved -> {save_path}")
    print(f"  window {a:.1f}-{b:.1f}s of {total:.0f}s sim | {speed:.0f}x speed | "
          f"{frames} frames @ {fps}fps = {playback:.1f}s clip | "
          f"{sspf*1000:.0f} ms sim/frame")
    if time_lapsed:
        print(f"  (time-lapse: hit max_frames={max_frames}, so it's sped up. For "
              f"real-time detail show a shorter window, e.g. t_end={a+15:.0f})")
    elif not full and b < total:
        print(f"  (showing first {b-a:.0f}s only. For the whole run: full=True)")

    # inline in a notebook
    try:
        from IPython.display import Image, Video, display
        display(Video(save_path, embed=True) if save_path.endswith('.mp4')
                else Image(filename=save_path))
    except Exception:
        pass
    return save_path


def animate_system(system, **kwargs):
    """
    Animate a System or ReactionWheelSystem directly -- works for both.

    Reads arm_thetas / arm_length off the object, and if it also has wheel_thetas
    / wheel_radius (a reaction wheel), draws the spinning wheel too. Pass any
    animate_pendulum kwarg (units, zero_ref, dt, speed, full, save_path, ...).

        animate_system(pendulum, units='deg', zero_ref='down', dt=DT)
        animate_system(rwp,       units='rad', zero_ref='down', dt=DT, full=True)
    """
    wheel_thetas = getattr(system, 'wheel_thetas', None)
    wheel_speeds = getattr(system, 'wheel_vthetas', None)   # for the speed readout
    if not wheel_thetas:                       # None or empty -> plain arm
        wheel_thetas, wheel_speeds = None, None
    return animate_pendulum(
        system.arm_thetas,
        arm_length=getattr(system, 'arm_length', 5.0),
        wheel_thetas=wheel_thetas,
        wheel_speeds=wheel_speeds,
        wheel_radius=getattr(system, 'wheel_radius', None),
        **kwargs,
    )
