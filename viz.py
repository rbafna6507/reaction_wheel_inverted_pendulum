"""
Pendulum visualizer.

Feed it a list of thetas (the angles you logged in your sim); it animates the
rod pivoting about the origin, saves a GIF (fast; pass save_path='x.mp4' for a
smaller/higher-quality MP4 instead), and shows it inline in a notebook.
It does NOT know any physics -- it draws the angles you give it, in order.

Rendering is done directly with Pillow (ImageDraw) rather than matplotlib: the
scene is a handful of primitives (rod, bob, wheel, spoke, trail, clock), so we
draw them straight onto a pre-rendered background in indexed-color mode. That's
~10x faster than matplotlib's FuncAnimation, which re-renders the whole figure
(axes/grid/ticks/fonts) every frame.

Works with BOTH systems:
  * plain arm      -> pass just the arm angles.
  * reaction wheel -> also pass the wheel angles; it draws the wheel at the arm
    tip with a spinning spoke, plus a wheel-SPEED readout (rad/s) to watch it
    saturate.

--------------------------------------------------------------------------------
SMOOTHNESS -- one paragraph, explains every knob
--------------------------------------------------------------------------------
Choppiness is set by how much SIM TIME passes between drawn frames:
    sim_seconds_per_frame = speed / fps          (low speed = smooth)
Knobs: `speed` (sim seconds per real playback second; 1 = real time),
`fps`, and the window (`t_start`..`t_end`, or `full=True`). You cannot show all
1000 s AND be smooth AND keep the clip short -- so the default shows a real-time
~20 s window; use full=True for a sped-up time-lapse of the whole run.

--------------------------------------------------------------------------------
Easiest calls
--------------------------------------------------------------------------------
    from viz import animate_system

    animate_system(pendulum, units='deg', zero_ref='down', dt=DT)   # plain arm
    animate_system(rwp,       units='rad', zero_ref='down', dt=DT)   # reaction wheel

By default you get a smooth, real-time GIF of the first ~20 s. To see the whole
run (time-lapse):  animate_system(rwp, ..., full=True)   (add save_path='x.mp4'
for a much smaller/higher-quality file if a long GIF gets big.)

`units` : 'deg' or 'rad'. `zero_ref` : 'down' (theta=0 hangs down; balancing at
pi means the rod ends up pointing UP) or 'up'.
"""

import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# indexed palette (index -> RGB); drawing in 'P' mode avoids per-frame quantize
_PAL_RGB = [
    (255, 255, 255),  # 0 background
    (225, 225, 225),  # 1 grid
    (200, 200, 200),  # 2 axes
    (58, 110, 165),   # 3 rod
    (193, 68, 14),    # 4 bob
    (0, 0, 0),        # 5 pivot
    (232, 178, 150),  # 6 trail
    (42, 42, 42),     # 7 wheel rim
    (224, 160, 48),   # 8 spoke / hub
    (30, 30, 30),     # 9 text
]
_PALETTE = [c for rgb in _PAL_RGB for c in rgb] + [0] * (768 - 3 * len(_PAL_RGB))


def _load_font(px):
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "Menlo.ttc", "cour.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            pass
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans Mono"), px)
    except Exception:
        return ImageFont.load_default()


def _tip_xy(theta_rad, L, zero_ref):
    """Position of the rod tip given the pivot at the origin."""
    if zero_ref == 'down':
        return L * math.sin(theta_rad), -L * math.cos(theta_rad)
    elif zero_ref == 'up':
        return -L * math.sin(theta_rad), L * math.cos(theta_rad)
    raise ValueError("zero_ref must be 'down' or 'up'")


def animate_pendulum(thetas, arm_length=5.0, units='deg', zero_ref='down',
                     dt=0.01, save_path='pendulum.gif',
                     fps=30, speed=1.0, t_start=0.0, t_end=None, full=False,
                     max_frames=6000, dpi=100, crf=18, trail=30,
                     wheel_thetas=None, wheel_speeds=None, wheel_radius=None):
    """
    Animate logged pendulum angles -> GIF (or MP4), shown inline in a notebook.

    Timing: `speed` (sim s per real s; 1=real time), `fps`, and the window
    (`t_start`/`t_end`, or `full=True` for the whole run). t_end=None -> a ~20 s
    real-time slice. `max_frames` caps frames (speed is auto-raised past it).
    Quality: `dpi` sets pixels (output = 5*dpi square; 100 -> 500px). `crf` is
    MP4 quality (lower=better). `trail` = # of past tip points drawn as a trace.
    Geometry: `arm_length`, `units` ('deg'/'rad'), `zero_ref` ('down'/'up').
    Wheel: `wheel_thetas` (spin angles) draws the wheel + spoke; `wheel_speeds`
    (rad/s) drives the readout (else derived from wheel_thetas); `wheel_radius`.

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

    # ---- choose the sim-time window and the frame count -----------------------
    total = (n - 1) * dt
    if full:
        a, b = 0.0, total
    elif t_end is None:
        a = max(0.0, t_start)
        b = min(total, a + 20.0)
    else:
        a = max(0.0, t_start)
        b = min(total, t_end)
    if b <= a:
        a, b = 0.0, total
    window = b - a

    # GIF frame delays live on a 10 ms grid (and most players clamp delays under
    # 20 ms up to 100 ms). Snap to a representable delay and use the ACTUAL rate
    # for all timing math -- otherwise e.g. fps=30 asks for 33 ms, the file stores
    # 30 ms, and the whole clip plays ~11% fast while the t= clock lies.
    is_gif = save_path.endswith('.gif')
    if is_gif:
        duration_cs = max(2, int(round(100.0 / fps)))     # centiseconds, >= 20 ms
        fps_eff = 100.0 / duration_cs
        if fps > 50:
            print(f"note: GIF can't play faster than 50 fps (20 ms/frame floor); "
                  f"using {fps_eff:.1f} fps. For higher rates use save_path='...mp4'.")
    else:
        duration_cs = None
        fps_eff = fps

    frames_n = max(2, int(round(window * fps_eff / speed)))
    time_lapsed = False
    if frames_n > max_frames:
        frames_n = max_frames
        speed = window * fps_eff / frames_n
        time_lapsed = True
    sspf = window / frames_n

    sim_t = np.linspace(a, b, frames_n)
    idx = np.clip(np.round(sim_t / dt).astype(int), 0, n - 1)
    theta_f = theta[idx]
    L = arm_length
    tips = np.array([_tip_xy(t, L, zero_ref) for t in theta_f])
    wtheta_f = wtheta[idx] if have_wheel else None
    wspeed_f = wspeed[idx] if have_wheel else None

    # ---- pixel geometry -------------------------------------------------------
    S = int(round(5 * dpi))
    if S % 2:
        S += 1                                   # even dims (h264 yuv420p)
    lim = 1.25 * L + (wheel_radius if have_wheel else 0.0)
    sc = S / (2.0 * lim)

    def to_px(x, y):
        return ((x + lim) * sc, (S - (y + lim) * sc))   # y up in physics -> down in image

    w_rod = max(2, round(0.008 * S))
    r_bob = max(3, round(0.028 * S))
    r_piv = max(2, round(0.016 * S))
    r_hub = max(2, round(0.012 * S))
    w_rim = max(1, round(0.004 * S))
    w_spoke = max(1, round(0.005 * S))
    w_trail = max(1, round(0.003 * S))
    font = _load_font(max(9, round(0.026 * S)))
    tx0, ty0 = round(0.03 * S), round(0.03 * S)

    # ---- pre-render the static background once --------------------------------
    bg = Image.new('P', (S, S), 0)
    bg.putpalette(_PALETTE)
    bd = ImageDraw.Draw(bg)
    g0, g1 = int(math.floor(-lim)), int(math.ceil(lim))
    for g in range(g0, g1 + 1):
        gx, _ = to_px(g, 0)
        bd.line([gx, 0, gx, S], fill=1, width=1)
        _, gy = to_px(0, g)
        bd.line([0, gy, S, gy], fill=1, width=1)
    ax0, ay0 = to_px(0, -lim)
    ax1, ay1 = to_px(0, lim)
    bd.line([ax0, ay0, ax1, ay1], fill=2, width=1)
    bx0, by0 = to_px(-lim, 0)
    bx1, by1 = to_px(lim, 0)
    bd.line([bx0, by0, bx1, by1], fill=2, width=1)

    ox, oy = to_px(0.0, 0.0)

    def render(i):
        img = bg.copy()
        d = ImageDraw.Draw(img)
        x, y = to_px(*tips[i])
        if trail:
            lo = max(0, i - trail)
            pts = [to_px(tx, ty) for tx, ty in tips[lo:i + 1]]
            if len(pts) >= 2:
                d.line([c for p in pts for c in p], fill=6, width=w_trail)
        if have_wheel:
            r = wheel_radius * sc
            d.ellipse([x - r, y - r, x + r, y + r], outline=7, width=w_rim)
            phi = wtheta_f[i]
            # smear arc: how far the spoke swept during THIS frame's slice of sim
            # time. Above ~pi rad/frame the spoke position is aliased (wagon-wheel
            # effect: a fast wheel looks slow/jerky), so the smear is the honest
            # speed indicator: longer arc = faster wheel; near-full ring = spinning
            # a whole turn (or more) per frame.
            sweep = float(wspeed_f[i]) * sspf
            if abs(sweep) > 0.15:
                sw = math.degrees(max(-2 * math.pi, min(2 * math.pi, sweep)))
                sw = max(-355.0, min(355.0, sw))
                a0 = -math.degrees(phi)              # PIL angles: y-down, clockwise
                start, end = (a0, a0 + sw) if sw > 0 else (a0 + sw, a0)
                rs = 0.82 * r
                d.arc([x - rs, y - rs, x + rs, y + rs], start=start, end=end,
                      fill=6, width=w_spoke)
            ex, ey = x + r * math.cos(phi), y - r * math.sin(phi)
            d.line([x, y, ex, ey], fill=8, width=w_spoke)
            d.ellipse([ex - r_hub, ey - r_hub, ex + r_hub, ey + r_hub], fill=8)
        d.line([ox, oy, x, y], fill=3, width=w_rod)
        d.ellipse([x - r_bob, y - r_bob, x + r_bob, y + r_bob], fill=4)
        d.ellipse([ox - r_piv, oy - r_piv, ox + r_piv, oy + r_piv], fill=5)
        txt = (f"t = {sim_t[i]:7.2f} s   ({speed:.0f}x)\n"
               f"theta = {math.degrees(theta_f[i]):7.2f} deg\n"
               f"({zero_ref}=0)")
        if have_wheel:
            w = wspeed_f[i]
            txt += f"\nwheel = {w:7.1f} rad/s ({w/(2*math.pi):+.1f} rev/s)"
        box = d.multiline_textbbox((tx0, ty0), txt, font=font, spacing=2)
        d.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3], fill=0)
        d.multiline_text((tx0, ty0), txt, font=font, fill=9, spacing=2)
        return img

    # ---- encode ---------------------------------------------------------------
    if is_gif:
        frames = [render(i) for i in range(frames_n)]
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=duration_cs * 10, loop=0, disposal=2,
                       optimize=False)
    elif save_path.endswith('.mp4'):
        proc = subprocess.Popen(
            ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{S}x{S}', '-r', str(fps), '-i', '-', '-an',
             '-vcodec', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p',
             save_path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        for i in range(frames_n):                # stream -> constant memory
            proc.stdin.write(render(i).convert('RGB').tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        raise ValueError("save_path must end in .gif or .mp4")

    playback = frames_n / fps_eff
    print(f"saved -> {save_path}")
    print(f"  window {a:.1f}-{b:.1f}s of {total:.0f}s sim | {speed:.0f}x speed | "
          f"{frames_n} frames @ {fps_eff:.4g}fps = {playback:.1f}s clip | "
          f"{sspf*1000:.0f} ms sim/frame")
    if have_wheel:
        peak_sweep = float(np.max(np.abs(wspeed_f))) * sspf
        if peak_sweep > 1.0:
            print(f"  (wheel sweeps up to {peak_sweep:.1f} rad per FRAME -- the spoke's "
                  f"apparent motion is aliased at this fps (wagon-wheel effect); judge "
                  f"speed by the smear arc + rad/s readout, not the spoke)")
    if is_gif and frames_n > 900:
        print(f"  ({frames_n}-frame GIFs are huge and stutter in browser playback; "
              f"for full runs prefer save_path='pendulum.mp4')")
    if time_lapsed:
        print(f"  (time-lapse: hit max_frames={max_frames}, so it's sped up. For "
              f"real-time detail show a shorter window, e.g. t_end={a+15:.0f})")
    elif not full and b < total:
        print(f"  (showing first {b-a:.0f}s only. For the whole run: full=True)")

    try:
        from IPython.display import Image as IPyImage, Video, display
        display(Video(save_path, embed=True) if save_path.endswith('.mp4')
                else IPyImage(filename=save_path))
    except Exception:
        pass
    return save_path


def animate_system(system, **kwargs):
    """
    Animate a System or ReactionWheelSystem directly -- works for both.

    Reads arm_thetas / arm_length off the object, and if it also has wheel_thetas
    / wheel_vthetas / wheel_radius (a reaction wheel) draws the wheel + a speed
    readout. Pass any animate_pendulum kwarg (units, zero_ref, dt, speed, full,
    save_path, ...).

        animate_system(pendulum, units='deg', zero_ref='down', dt=DT)
        animate_system(rwp,       units='rad', zero_ref='down', dt=DT, full=True)
    """
    wheel_thetas = getattr(system, 'wheel_thetas', None)
    wheel_speeds = getattr(system, 'wheel_vthetas', None)
    if not wheel_thetas:
        wheel_thetas, wheel_speeds = None, None
    return animate_pendulum(
        system.arm_thetas,
        arm_length=getattr(system, 'arm_length', 5.0),
        wheel_thetas=wheel_thetas,
        wheel_speeds=wheel_speeds,
        wheel_radius=getattr(system, 'wheel_radius', None),
        **kwargs,
    )
