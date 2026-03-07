"""Motor trace decoder: trajectory extraction, GRU decoder, differentiable renderer."""
import torch
import torch.nn as nn
import os
import math
import numpy as np


# --- Trajectory Extraction (fonttools) ---

def _flatten_cubic(p0, p1, p2, p3, n_segments=4):
    """Flatten a cubic bezier curve to line segments.

    Args:
        p0..p3: control points as (x, y) tuples
        n_segments: number of line segments to approximate the curve
    Returns:
        list of (x, y) tuples (n_segments points, excluding p0)
    """
    points = []
    for i in range(1, n_segments + 1):
        t = i / n_segments
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        points.append((x, y))
    return points


def _resample_trajectory(points, pen_states, n_points):
    """Resample a trajectory to exactly n_points via arc-length parameterization.

    Args:
        points: list of (x, y) tuples
        pen_states: list of float (0.0=move, 1.0=stroke), same length as points
        n_points: target number of points
    Returns:
        (resampled_points, resampled_pen_states)
    """
    if len(points) <= 1:
        pt = points[0] if points else (0.0, 0.0)
        return [pt] * n_points, [1.0] * n_points

    # Compute cumulative arc lengths
    cum_len = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cum_len.append(cum_len[-1] + math.sqrt(dx * dx + dy * dy))

    total_len = cum_len[-1]
    if total_len < 1e-8:
        return [points[0]] * n_points, [pen_states[0]] * n_points

    resampled = []
    resampled_pen = []
    j = 0
    for i in range(n_points):
        target = i * total_len / max(n_points - 1, 1)
        while j < len(cum_len) - 1 and cum_len[j + 1] < target:
            j += 1
        if j >= len(cum_len) - 1:
            resampled.append(points[-1])
            resampled_pen.append(pen_states[-1])
        else:
            seg_len = cum_len[j + 1] - cum_len[j]
            if seg_len < 1e-8:
                t = 0.0
            else:
                t = (target - cum_len[j]) / seg_len
            x = points[j][0] + t * (points[j + 1][0] - points[j][0])
            y = points[j][1] + t * (points[j + 1][1] - points[j][1])
            resampled.append((x, y))
            # Pen state: use the state of the segment we're interpolating within
            resampled_pen.append(pen_states[j + 1])

    return resampled, resampled_pen


def _fallback_trajectory(n_points):
    """Fallback trajectory for missing glyphs: centered dot."""
    return torch.zeros(n_points, 3)


# --- Centerline trajectory extraction ---

def _render_letter_binary(font_path, char, img_size=128, font_size=60):
    """Render a letter as a binary numpy array (1=ink, 0=background)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('L', (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)
    if font_path is None:
        font = ImageFont.load_default(size=font_size)
    else:
        font = ImageFont.truetype(font_path, size=font_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (img_size - bbox[2] - bbox[0]) / 2
    y = (img_size - bbox[3] - bbox[1]) / 2
    draw.text((x, y), char, fill=255, font=font)

    return (np.array(img) > 127).astype(np.uint8)


def _skeletonize(binary):
    """Zhang-Suen morphological thinning — produces clean, connected 1px skeleton."""


    img = binary.copy().astype(np.uint8)
    rows, cols = img.shape

    def _neighbors(img, r, c):
        """Return 8-neighbors in clockwise order: P2,P3,P4,P5,P6,P7,P8,P9."""
        return [
            img[r-1, c],   img[r-1, c+1], img[r, c+1],   img[r+1, c+1],
            img[r+1, c],   img[r+1, c-1], img[r, c-1],   img[r-1, c-1],
        ]

    def _transitions(neighbors):
        """Count 0->1 transitions in the circular sequence."""
        n = neighbors + [neighbors[0]]
        return sum(1 for i in range(8) if n[i] == 0 and n[i+1] == 1)

    changed = True
    while changed:
        changed = False

        # Sub-iteration 1
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                nb = _neighbors(img, r, c)
                B = sum(nb)
                A = _transitions(nb)
                if (2 <= B <= 6 and A == 1 and
                    nb[0] * nb[2] * nb[4] == 0 and
                    nb[2] * nb[4] * nb[6] == 0):
                    to_remove.append((r, c))
        for r, c in to_remove:
            img[r, c] = 0
            changed = True

        # Sub-iteration 2
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                nb = _neighbors(img, r, c)
                B = sum(nb)
                A = _transitions(nb)
                if (2 <= B <= 6 and A == 1 and
                    nb[0] * nb[2] * nb[6] == 0 and
                    nb[0] * nb[4] * nb[6] == 0):
                    to_remove.append((r, c))
        for r, c in to_remove:
            img[r, c] = 0
            changed = True

    return img


def _order_skeleton_pixels(skel):
    """Order skeleton pixels into a pen trajectory via graph-based stroke tracing.

    Builds an adjacency graph from the skeleton, identifies endpoints and
    junctions, then traces connected strokes. Pen lifts only between
    disconnected strokes.

    Returns list of (x, y, pen_down) tuples.
    """


    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    N = len(xs)
    # Map (y, x) -> index for fast neighbor lookup
    coord_to_idx = {}
    for i in range(N):
        coord_to_idx[(ys[i], xs[i])] = i

    # Build 8-connected adjacency lists
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        y, x = ys[i], xs[i]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = coord_to_idx.get((y + dy, x + dx))
                if j is not None:
                    neighbors[i].append(j)

    # Classify pixels: degree = number of neighbors
    degree = np.array([len(neighbors[i]) for i in range(N)])
    endpoints = np.where(degree == 1)[0]  # dead ends

    # Choose starting points: prefer endpoints, sorted top-left
    visited = np.zeros(N, dtype=bool)
    ordered = []

    def _pick_start():
        """Pick best unvisited starting point: endpoint first, then any."""
        unvisited_endpoints = [e for e in endpoints if not visited[e]]
        if unvisited_endpoints:
            # Top-left preference
            best = min(unvisited_endpoints, key=lambda i: (ys[i], xs[i]))
            return best
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            return None
        return min(unvisited, key=lambda i: (ys[i], xs[i]))

    def _trace_from(start_idx):
        """Trace a stroke from start_idx, following unvisited neighbors.
        At junctions, prefer the straightest continuation."""
        chain = [start_idx]
        visited[start_idx] = True
        current = start_idx

        while True:
            # Find unvisited neighbors
            unvis = [j for j in neighbors[current] if not visited[j]]
            if not unvis:
                break
            if len(unvis) == 1:
                nxt = unvis[0]
            else:
                # At junction: prefer straightest continuation
                if len(chain) >= 2:
                    prev = chain[-2]
                    dx0 = xs[current] - xs[prev]
                    dy0 = ys[current] - ys[prev]
                    best_dot = -2
                    nxt = unvis[0]
                    for j in unvis:
                        dx1 = xs[j] - xs[current]
                        dy1 = ys[j] - ys[current]
                        # Cosine similarity (unnormalized, just for comparison)
                        dot = dx0 * dx1 + dy0 * dy1
                        if dot > best_dot:
                            best_dot = dot
                            nxt = j
                else:
                    # No direction history — pick topmost/leftmost
                    nxt = min(unvis, key=lambda j: (ys[j], xs[j]))

            visited[nxt] = True
            chain.append(nxt)
            current = nxt

        return chain

    while True:
        start = _pick_start()
        if start is None:
            break
        chain = _trace_from(start)
        pen_start = 0.0 if len(ordered) > 0 else 1.0  # pen up between strokes
        for k, idx in enumerate(chain):
            pen = pen_start if k == 0 else 1.0
            ordered.append((float(xs[idx]), float(ys[idx]), pen))

    return ordered


def extract_centerline_trajectory(font_path, char, n_points=32, img_size=128):
    """Extract a centerline (stroke-center) trajectory for a character.

    Unlike extract_glyph_trajectory which traces font vector outlines,
    this renders the letter, skeletonizes it to 1px-wide strokes,
    then orders the skeleton pixels into a pen trajectory.

    Returns: (n_points, 3) tensor of (x, y, pen_down) in [-1, 1].
    """
    import scipy.ndimage as ndi

    binary = _render_letter_binary(font_path, char, img_size=img_size)
    binary = ndi.binary_closing(binary, structure=np.ones((3, 3))).astype(np.uint8)
    skel = _skeletonize(binary)

    ordered = _order_skeleton_pixels(skel)
    if len(ordered) < 2:
        return _fallback_trajectory(n_points)

    # Convert pixel coords to [-1, 1] (centered, uniform scaling)
    points = [(p[0], p[1]) for p in ordered]
    pen_states = [p[2] for p in ordered]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # Pixel coords: center of image = img_size/2
    # Map to [-1, 1]: x_norm = (x - center) / (img_size/2)
    center = img_size / 2.0
    half = img_size / 2.0
    norm_points = [((p[0] - center) / half, (p[1] - center) / half) for p in points]

    # Resample to n_points
    resampled, resampled_pen = _resample_trajectory(norm_points, pen_states, n_points)

    trajectory = torch.zeros(n_points, 3)
    for i, ((x, y), pen) in enumerate(zip(resampled, resampled_pen)):
        trajectory[i, 0] = x
        trajectory[i, 1] = y
        trajectory[i, 2] = pen

    return trajectory


def extract_glyph_trajectory(font_path, char, n_points=32):
    """TTF glyph -> (n_points, 3) tensor of (x, y, pen_down).

    Uses fonttools RecordingPen to get moveTo/lineTo/curveTo operations.
    Flattens bezier curves to line segments.
    Resamples to exactly n_points via arc-length parameterization.
    Normalizes to [-1, 1] coordinate system (matching grid_sample).
    pen_down: 0.0 for moves between contours, 1.0 for strokes.
    """
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import RecordingPen

    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()

    # Map character to glyph name
    cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(char))
    if glyph_name is None or glyph_name not in glyph_set:
        return _fallback_trajectory(n_points)

    pen = RecordingPen()
    glyph_set[glyph_name].draw(pen)

    if not pen.value:
        return _fallback_trajectory(n_points)

    # Convert pen operations to point sequences
    points = []
    pen_states = []
    current = (0.0, 0.0)

    for op, args in pen.value:
        if op == 'moveTo':
            current = args[0]
            points.append(current)
            pen_states.append(0.0)  # move = pen up
        elif op == 'lineTo':
            pt = args[0]
            points.append(pt)
            pen_states.append(1.0)  # stroke
            current = pt
        elif op == 'curveTo':
            # Cubic bezier: current, cp1, cp2, endpoint
            if len(args) == 3:
                segs = _flatten_cubic(current, args[0], args[1], args[2])
                for s in segs:
                    points.append(s)
                    pen_states.append(1.0)
                current = args[2]
            elif len(args) == 2:
                # Quadratic bezier approximated as cubic
                cp = args[0]
                end = args[1]
                # Convert quadratic to cubic control points
                cp1 = (current[0] + 2/3 * (cp[0] - current[0]),
                        current[1] + 2/3 * (cp[1] - current[1]))
                cp2 = (end[0] + 2/3 * (cp[0] - end[0]),
                        end[1] + 2/3 * (cp[1] - end[1]))
                segs = _flatten_cubic(current, cp1, cp2, end)
                for s in segs:
                    points.append(s)
                    pen_states.append(1.0)
                current = end
        elif op == 'qCurveTo':
            # TrueType quadratic splines — may have implied on-curve points
            for k in range(len(args)):
                if k == len(args) - 1:
                    end = args[k]
                else:
                    # Implied on-curve point between consecutive off-curve
                    end = ((args[k][0] + args[k + 1][0]) / 2,
                           (args[k][1] + args[k + 1][1]) / 2) if k < len(args) - 2 else args[k + 1]
                cp = args[k] if k < len(args) - 1 else current
                # Quadratic to cubic
                cp1 = (current[0] + 2/3 * (cp[0] - current[0]),
                        current[1] + 2/3 * (cp[1] - current[1]))
                cp2 = (end[0] + 2/3 * (cp[0] - end[0]),
                        end[1] + 2/3 * (cp[1] - end[1]))
                segs = _flatten_cubic(current, cp1, cp2, end)
                for s in segs:
                    points.append(s)
                    pen_states.append(1.0)
                current = end
                if k == len(args) - 1:
                    break
        elif op == 'closePath' or op == 'endPath':
            if points and len(points) >= 2:
                # Close by drawing back to the start of this contour
                # Find the last moveTo
                for idx in range(len(pen_states) - 1, -1, -1):
                    if pen_states[idx] == 0.0:
                        points.append(points[idx])
                        pen_states.append(1.0)
                        break

    if not points:
        return _fallback_trajectory(n_points)

    # Normalize to [-1, 1]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0
    # Use uniform scaling to preserve aspect ratio
    scale = max(range_x, range_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    norm_points = [((p[0] - cx) / scale * 2, (p[1] - cy) / scale * 2)
                   for p in points]
    # Flip y: font coordinates have y-up, our grid has y-down
    norm_points = [(p[0], -p[1]) for p in norm_points]

    # Resample
    resampled, resampled_pen = _resample_trajectory(norm_points, pen_states, n_points)

    # Build tensor
    trajectory = torch.zeros(n_points, 3)
    for i, ((x, y), pen) in enumerate(zip(resampled, resampled_pen)):
        trajectory[i, 0] = x
        trajectory[i, 1] = y
        trajectory[i, 2] = pen

    return trajectory


# --- Font Path Resolution ---

FONT_PATHS = {
    'dejavu-sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
}


def resolve_font_path(font_name):
    """Resolve a font name to a file path."""
    if font_name in FONT_PATHS:
        return FONT_PATHS[font_name]
    if os.path.exists(font_name):
        return font_name
    raise ValueError(f"Unknown font: {font_name}. Available: {list(FONT_PATHS.keys())}")


# --- Pre-generation ---

def generate_trajectory_dataset(output_dir, font_name='dejavu-sans', n_points=32,
                                 letters=None, mode='outline'):
    """Pre-generate all letter trajectories for canonical font.

    Args:
        mode: 'outline' (font vector contours) or 'centerline' (skeletonized strokes)

    Saves: output_dir/trajectories.pt  (dict: char -> (N, 3) tensor)
    Saves: output_dir/trajectory_atlas.png  (visualization grid)
    """
    import matplotlib.pyplot as plt

    if letters is None:
        letters = ([chr(i) for i in range(65, 91)] +
                   [chr(i) for i in range(97, 123)])

    font_path = resolve_font_path(font_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {mode} trajectories for {len(letters)} letters ({font_name})")

    traj_dict = {}
    for char in letters:
        if mode == 'centerline':
            traj = extract_centerline_trajectory(font_path, char, n_points=n_points)
        else:
            traj = extract_glyph_trajectory(font_path, char, n_points=n_points)
        traj_dict[char] = traj
        pen_up = (traj[:, 2] < 0.5).sum().item()
        pen_down = (traj[:, 2] >= 0.5).sum().item()
        print(f"  {char}: {n_points} points, {pen_down} stroke / {pen_up} move")

    # Save
    save_path = os.path.join(output_dir, 'trajectories.pt')
    torch.save(traj_dict, save_path)
    print(f"Saved {len(traj_dict)} trajectories to {save_path}")

    # Visualization atlas
    n_chars = len(letters)
    n_cols = 13
    n_rows = (n_chars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, char in enumerate(letters):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        traj = traj_dict[char]
        xs = traj[:, 0].numpy()
        ys = traj[:, 1].numpy()
        pen = traj[:, 2].numpy()

        # Draw stroke segments
        for j in range(1, len(xs)):
            if pen[j] > 0.5:
                ax.plot([xs[j - 1], xs[j]], [ys[j - 1], ys[j]],
                        'b-', linewidth=1.0)
            else:
                ax.plot([xs[j - 1], xs[j]], [ys[j - 1], ys[j]],
                        'r:', linewidth=0.3, alpha=0.3)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.invert_yaxis()  # y-down to match image coordinates
        ax.set_aspect('equal')
        ax.set_title(char, fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for i in range(n_chars, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')

    plt.tight_layout()
    atlas_path = os.path.join(output_dir, 'trajectory_atlas.png')
    plt.savefig(atlas_path, dpi=150)
    plt.close()
    print(f"Atlas saved to {atlas_path}")


# --- Runtime lookup ---

def load_trajectory_data(trajectory_dir):
    """Load pre-generated trajectory data from disk.

    Returns dict: char -> (N, 3) tensor.
    """
    path = os.path.join(trajectory_dir, 'trajectories.pt')
    return torch.load(path, map_location='cpu', weights_only=True)


def batch_gt_trajectories(letters, cases, traj_data, device):
    """Build (B, N, 3) ground truth from per-character trajectory dict.

    Args:
        letters: list of B uppercase letter chars ('A'..'Z')
        cases: list of B case strings ('upper' or 'lower')
        traj_data: dict from load_trajectory_data()
        device: torch device
    Returns:
        (B, N, 3) tensor
    """
    batch = []
    for letter, case in zip(letters, cases):
        if case == 'lower':
            char = letter.lower()
        else:
            char = letter.upper()
        if char in traj_data:
            batch.append(traj_data[char])
        else:
            # Fallback
            n_points = next(iter(traj_data.values())).shape[0]
            batch.append(torch.zeros(n_points, 3))
    return torch.stack(batch).to(device)


# --- Motor Trace Decoder ---

class MotorTraceDecoder(nn.Module):
    """latent (B, 256) -> trajectory (B, N, 3) = (x, y, pen_down_logit)

    GRU-based autoregressive decoder. Stroke points are naturally sequential.
    Architecture: latent -> Linear -> GRU h0, then unroll N steps.
    Each step: GRU(prev_point, h) -> point_head -> (x, y), pen_head -> logit
    """
    def __init__(self, latent_dim=256, hidden_dim=256, n_points=32):
        super().__init__()
        self.n_points = n_points
        self.latent_to_h = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRUCell(3, hidden_dim)
        self.point_head = nn.Linear(hidden_dim, 2)    # tanh -> [-1, 1]
        self.pen_head = nn.Linear(hidden_dim, 1)       # logit
        self.start_token = nn.Parameter(torch.zeros(1, 3))

    def forward(self, latent):
        """Decode latent to trajectory.

        Args:
            latent: (B, latent_dim)
        Returns:
            trajectory: (B, N, 3) where [:,:,:2] = tanh(xy), [:,:,2] = pen logit
        """
        B = latent.shape[0]
        h = torch.tanh(self.latent_to_h(latent))  # (B, hidden_dim)
        inp = self.start_token.expand(B, -1)  # (B, 3)

        points = []
        for _ in range(self.n_points):
            h = self.gru(inp, h)
            xy = torch.tanh(self.point_head(h))       # (B, 2)
            pen = self.pen_head(h)                     # (B, 1)
            point = torch.cat([xy, pen], dim=1)        # (B, 3)
            points.append(point)
            inp = point.detach()  # autoregressive: feed predicted point back
            # Detach to prevent backprop through all previous steps
            # (teacher forcing equivalent for stability)

        return torch.stack(points, dim=1)  # (B, N, 3)


# --- Differentiable Soft Renderer ---

def soft_render(trajectory, height=128, width=128, sigma=1.5):
    """Differentiable rendering: Gaussian blobs along trajectory.

    VRAM-efficient: loops over N points, never materializes (B, N, H, W).
    Per iteration: (B, 1, H, W) blob, weighted by sigmoid(pen_down).
    Accumulates into canvas. ~3MB per iteration at B=52.

    Args:
        trajectory: (B, N, 3) where [:,:,:2] are xy in [-1,1], [:,:,2] are pen logits
        height, width: output image size
        sigma: Gaussian blob width in pixels
    Returns:
        canvas: (B, 1, H, W) rendered image, values in [0, 1]
    """
    B, N, _ = trajectory.shape
    device = trajectory.device

    # Pre-compute coordinate grids
    gy = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1)
    gx = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width)

    # Sigma in normalized coordinates
    sigma_norm_h = sigma * 2.0 / height
    sigma_norm_w = sigma * 2.0 / width

    canvas = torch.zeros(B, 1, height, width, device=device)

    for t in range(N):
        x = trajectory[:, t, 0].view(B, 1, 1, 1)  # (B, 1, 1, 1)
        y = trajectory[:, t, 1].view(B, 1, 1, 1)
        pen_logit = trajectory[:, t, 2].view(B, 1, 1, 1)

        dx = (gx - x) / sigma_norm_w
        dy = (gy - y) / sigma_norm_h
        blob = torch.exp(-0.5 * (dx * dx + dy * dy))  # (B, 1, H, W)
        canvas = canvas + torch.sigmoid(pen_logit) * blob

    return canvas.clamp(0, 1)


# --- Constrained Motor Decoder (v5) ---

class ConstrainedMotorDecoder(nn.Module):
    """latent -> K gated strokes of N points (xy only, no pen state).

    Each stroke has a latent-predicted start point. A GRU unrolls N points
    per stroke, with hidden state carrying across strokes for inter-stroke
    context. Per-stroke sigmoid gates allow unused strokes to produce zero ink.
    """
    def __init__(self, latent_dim=256, hidden_dim=256,
                 n_strokes=4, points_per_stroke=20):
        super().__init__()
        self.n_strokes = n_strokes
        self.points_per_stroke = points_per_stroke
        self.latent_to_h = nn.Linear(latent_dim, hidden_dim)
        self.stroke_start_head = nn.Linear(latent_dim, n_strokes * 2)
        self.gru = nn.GRUCell(2, hidden_dim)
        self.point_head = nn.Linear(hidden_dim, 2)
        self.gate_head = nn.Linear(latent_dim, n_strokes)

    def forward(self, latent):
        """Decode latent to gated stroke points.

        Args:
            latent: (B, latent_dim)
        Returns:
            points: (B, K, N, 2) stroke points in [-1, 1]
            gates: (B, K) per-stroke activation in [0, 1]
        """
        B = latent.shape[0]
        K = self.n_strokes
        N = self.points_per_stroke

        h = torch.tanh(self.latent_to_h(latent))
        starts = torch.tanh(self.stroke_start_head(latent)).view(B, K, 2)
        gates = torch.sigmoid(self.gate_head(latent))

        all_points = []
        for s in range(K):
            inp = starts[:, s]
            for t in range(N):
                h = self.gru(inp, h)
                xy = torch.tanh(self.point_head(h))
                all_points.append(xy)
                inp = xy.detach()  # autoregressive

        points = torch.stack(all_points, dim=1).view(B, K, N, 2)
        return points, gates


def render_gated_strokes(points, gates, height=128, width=128, sigma=0.75):
    """Render gated strokes as Gaussian blobs on a canvas.

    Args:
        points: (B, K, N, 2) stroke points in [-1, 1]
        gates: (B, K) per-stroke activation in [0, 1]
        height, width: output canvas size
        sigma: Gaussian blob width in pixels
    Returns:
        canvas: (B, 1, H, W) rendered image, clamped to [0, 1]
    """
    B, K, N, _ = points.shape
    device = points.device

    gy = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1)
    gx = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width)

    sigma_norm_h = sigma * 2.0 / height
    sigma_norm_w = sigma * 2.0 / width

    canvas = torch.zeros(B, 1, height, width, device=device)

    for k in range(K):
        gate = gates[:, k].view(B, 1, 1, 1)
        for t in range(N):
            x = points[:, k, t, 0].view(B, 1, 1, 1)
            y = points[:, k, t, 1].view(B, 1, 1, 1)
            dx = (gx - x) / sigma_norm_w
            dy = (gy - y) / sigma_norm_h
            blob = torch.exp(-0.5 * (dx * dx + dy * dy))
            canvas = canvas + gate * blob

    return canvas.clamp(0, 1)
