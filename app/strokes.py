"""
Turns a photo of a handwritten sticky note into ordered pen strokes for the wall's neon writing.

Pipeline (classical image processing, no AI model, so it works for any script):
  1. Find the note (its outline, else its colour) and crop to the paper.
  2. Ink mask: flatten the paper's lighting, then threshold with hysteresis, so faint pen that
     touches strong pen is kept and shadows are not.
  3. Find the ruled lines. One of them is the divider: writing above it is the NAME, writing below
     it is the MESSAGE. Ruled lines (and printed labels like "Name:") are erased from the ink.
  4. Thin the ink to 1-pixel centre lines (Zhang-Suen), trace them into polylines, drop tiny
     spurs, join pieces that touch, smooth (Chaikin) and resample evenly.
  5. Order strokes like writing (row by row, left to right) and normalise.

Output coordinates: x in 0..1 across the note, y in 0..aspect (top = 0), aspect = height / width.
Each stroke is {"p": [x0, y0, x1, y1, ...], "g": 1 for the name, 0 for the message}.
No text is read and nothing is stored here.
"""
import cv2
import numpy as np

WORK_WIDTH = 1600           # processing width: pen lines stay several pixels wide, so they do not break up
MIN_STROKE_FRAC = 0.0025    # drop specks shorter than this fraction of the note width
SPUR_FRAC = 0.006           # drop side branches shorter than this fraction of the note width
JOIN_FRAC = 0.003           # join stroke ends closer than this fraction of the note width
SPACING_FRAC = 0.004        # point spacing after resampling, as a fraction of the note width


def extract(jpeg, label_margin=0.22, stages=False):
    """Returns {"aspect", "divider", "strokes"} or raises ValueError. divider is the y of the name/message line (or None).
    With stages=True the result also has "stages": the images after each step (for the debug window):
      raw (the cropped note), bw (black and white ink), clean (after hole filling, gap bridging, rule removal).
    """
    data = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image")

    note, paper = _isolate_note(image)
    h, w = note.shape[:2]
    bw = _ink_mask(note, paper)                  # step 1: raw photo -> black and white ink
    ink = _fill_pen_holes(bw)                     # step 2: clean the ink: holes inside pen dots, ragged edges

    rules = find_rules(ink)
    divider = _pick_divider(ink, rules)
    cleaned = _remove_rules_and_labels(ink, rules, label_margin)
    cleaned = _drop_specks(cleaned)

    skeleton = _thin(cleaned > 0)
    pieces = _trace(skeleton, w)
    pieces = [s for s in pieces if _keep(s, w, h)]
    pieces = [_smooth(s, w) for s in pieces]

    name, message = _split(pieces, divider)
    name_ordered = _order(name, h)
    message_ordered = _order(message, h)
    ordered = name_ordered + message_ordered
    groups = [1] * len(name_ordered) + [0] * len(message_ordered)

    out = []
    for s, g in zip(ordered, groups):
        flat = []
        for x, y in s:
            flat.append(round(float(x) / w, 4))
            flat.append(round(float(y) / w, 4))     # y is normalised by width too: keeps proportions
        out.append({"p": flat, "g": g})
    div_y = None
    if divider is not None:
        div_y = round(float((divider[0] + divider[1]) / 2.0) / w, 4)
    result = {"aspect": round(h / float(w), 4), "divider": div_y, "strokes": out, "note_size": (w, h)}
    if stages:
        result["stages"] = {"raw": note, "bw": bw, "clean": cleaned}
    return result


# ---------------------------------------------------------------------- note

def _isolate_note(image):
    """Crops to the sticky note and returns (note image, paper mask)."""
    scale = WORK_WIDTH / float(image.shape[1])
    image = cv2.resize(image, (WORK_WIDTH, int(round(image.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    quad = _find_note_outline(image)
    if quad is not None:
        return _crop_to(image, quad)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centre = hsv[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5].reshape(-1, 3)
    hue, sat, val = [float(np.median(centre[:, i])) for i in range(3)]
    dh = np.abs(hsv[..., 0].astype(np.int16) - int(hue))
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(hsv[..., 1].astype(np.int16) - int(sat))
    bright_enough = hsv[..., 2].astype(np.int16) >= val - 60
    if sat < 40:
        paper = (hsv[..., 1] < 60) & bright_enough
    else:
        paper = (dh <= 6) & (ds <= 35) & bright_enough
    paper = paper.astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, np.full((h, w), 255, np.uint8)
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.12 * h * w:
        return image, np.full((h, w), 255, np.uint8)
    hull = cv2.convexHull(biggest)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return _crop_to(image, quad if len(quad) >= 4 else hull)


def _find_note_outline(image):
    """The note's own outline: the largest convex four-sided shape with strong edges."""
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.12 * h * w
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area <= best_area or area > 0.97 * h * w:
            continue
        quad = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            best, best_area = quad, area
    return best


def _crop_to(image, polygon):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [polygon], -1, 255, -1)
    # Inside the outline, keep only paper-coloured pixels: the desk can show at an edge, and its
    # light wood grain is bright enough to pass a brightness test but is the wrong colour.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    inside = hsv[mask > 0]
    hue, sat, val = [float(np.median(inside[:, i])) for i in range(3)]
    dh = np.abs(hsv[..., 0].astype(np.int16) - int(hue))
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(hsv[..., 1].astype(np.int16) - int(sat))
    if sat < 40:
        colour = (hsv[..., 1] < 70) & (hsv[..., 2] >= val * 0.6)
    else:
        colour = (dh <= 8) & (ds <= 75) & (hsv[..., 2] >= val * 0.55)
    colour = cv2.bitwise_and(colour.astype(np.uint8) * 255, mask)
    colour = cv2.morphologyEx(colour, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    # The paper is the large paper-coloured region(s). Choosing them BEFORE closing the writing holes keeps
    # the desk, the pen and the paper's edge shadow (separate, or only linked by a dark line) out.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colour, connectivity=4)
    if count > 1:
        # Keep every sizeable region, not just the largest: a divider line drawn across the note
        # cuts the paper in two, and both halves are paper. Desk fragments are far smaller.
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= 0.2 * areas.max())[0]
        colour = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
    # Fill the writing: any region of the paper mask that is enclosed by paper is paper with writing on it,
    # however wide the word (a fixed closing kernel erased long connected words).
    # First bridge thin dark lines (a divider or ruled line is still paper-side writing), then fill.
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    outlines, _ = cv2.findContours(colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colour = np.zeros_like(colour)
    cv2.drawContours(colour, outlines, -1, 255, -1)
    # If the colour test rejects a large part of the outline it has failed (strong or uneven lighting):
    # trust the outline alone rather than throw the writing away.
    if cv2.countNonZero(cv2.bitwise_and(mask, colour)) >= 0.7 * cv2.countNonZero(mask):
        mask = cv2.bitwise_and(mask, colour)
    # Stay away from the paper edge: its outline and shadow are not writing.
    inset = max(8, int(0.035 * w))
    mask = cv2.erode(mask, np.ones((inset, inset), np.uint8))
    x, y, bw, bh = cv2.boundingRect(mask)
    return image[y:y + bh, x:x + bw], mask[y:y + bh, x:x + bw]


# ---------------------------------------------------------------------- ink

def _ink_mask(note, paper):
    """Ink = clearly darker than the local paper. Hysteresis keeps faint strokes that touch strong ink."""
    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(31, (w // 16) | 1)
    # Paper brightness under the writing: a big closing removes thin dark strokes, a blur smooths it.
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 5.0)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)
    strong = (norm < 0.70) & (paper > 0)
    weak = (norm < 0.84) & (paper > 0)
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    ink = np.isin(labels, keep)
    ink = ink.astype(np.uint8) * 255
    # Bridge hairline breaks and shave ragged edges (which would grow spurs when thinned).
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return ink


def _fill_pen_holes(ink):
    """Fills tiny holes inside the ink (a pen dot with a pinhole would otherwise thin to a ring) and
    smooths ragged edges with a light open/close that keeps even thin pen lines whole."""
    h, w = ink.shape
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        max_hole = (0.006 * w) ** 2
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] != -1 and cv2.contourArea(c) < max_hole:      # a hole, and a small one
                cv2.drawContours(ink, [c], -1, 255, -1)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return ink


def _drop_specks(ink):
    """Removes ink blobs too small to be writing (paper texture, dust, leftover rule crumbs)."""
    h, w = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(20, int(w * w * 0.00005))
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], ink, 0).astype(np.uint8)


# ---------------------------------------------------------------------- rules

def find_rules(ink):
    """Ruled lines as (y at x start, y at x end, x start, x end, wide), top to bottom.

    Hand-drawn rules slope and break into pieces. Each straight piece is extended to the note's
    centre column; pieces landing at the same height belong to one rule, which gets a straight-line
    fit. Rows of letters also produce straight pieces, so a rule must be solid ink along its length,
    and (unless it is a wide divider) nearly empty just below it: a Devanagari word hangs from a
    solid top bar that otherwise looks exactly like a rule.
    """
    h, w = ink.shape
    segments = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=max(50, w // 14),
                               minLineLength=max(40, w // 10), maxLineGap=max(14, w // 50))
    if segments is None:
        return []
    mid = w / 2.0
    pieces = []
    for x0, y0, x1, y1 in segments[:, 0].astype(float):
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if x1 - x0 < 1 or abs(y1 - y0) > 0.12 * (x1 - x0):
            continue
        slope = (y1 - y0) / (x1 - x0)
        pieces.append((y0 + slope * (mid - x0), x0, y0, x1, y1))
    pieces.sort()
    clusters, tol = [], max(10.0, h * 0.015)
    for p in pieces:
        if clusters and p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    # Measured on the samples: real rules cover 0.88-1.0 of their length, rows of letters <= 0.73.
    k = max(7, (w // 170) | 1)
    thick = cv2.dilate(ink, np.ones((k, k), np.uint8))
    rules = []
    for c in clusters:
        xs = np.array([v for p in c for v in (p[1], p[3])])
        ys = np.array([v for p in c for v in (p[2], p[4])])
        if xs.max() - xs.min() < 0.3 * w:
            continue
        slope, icept = np.polyfit(xs, ys, 1) if len(set(xs)) > 1 else (0.0, ys.mean())
        x0, x1 = xs.min(), xs.max()
        # Judge the line by its middle 80%: hand-drawn rules hook at the ends.
        margin = 0.1 * (x1 - x0)
        sx = np.linspace(x0 + margin, x1 - margin, num=max(2, int(x1 - x0 - 2 * margin)))
        sy = slope * sx + icept
        hits = thick[np.clip(sy.astype(int), 0, h - 1), np.clip(sx.astype(int), 0, w - 1)] > 0
        if hits.mean() < 0.82:
            continue
        wide = (x1 - x0) >= 0.6 * w
        if not wide:
            below = np.mean([(ink[np.clip((sy + d).astype(int), 0, h - 1), np.clip(sx.astype(int), 0, w - 1)] > 0).mean()
                             for d in range(10, max(12, int(0.04 * h)))])
            if below > 0.08:
                continue
        rules.append((slope * x0 + icept, slope * x1 + icept, x0, x1, wide))
    rules.sort(key=lambda r: (r[0] + r[1]) / 2)
    return rules


def _ink_near(ink, rule, above, band):
    """Amount of ink in a band just above or below a rule."""
    h, w = ink.shape
    y0, y1, x0, x1 = rule[:4]
    sx = np.linspace(x0, x1, num=max(2, int(x1 - x0))).astype(int)
    sy = np.interp(sx, [x0, x1], [y0, y1])
    total = 0
    for d in range(6, band, 3):
        yy = np.clip((sy - d if above else sy + d).astype(int), 0, h - 1)
        total += int((ink[yy, np.clip(sx, 0, w - 1)] > 0).sum())
    return total


def _pick_divider(ink, rules):
    """The rule that splits name (above) from message (below): the first one with writing on both sides."""
    if not rules:
        return None
    h, w = ink.shape
    band = max(30, int(0.12 * h))
    need = 25
    for rule in rules:
        if _ink_near(ink, rule, True, band) >= need and _ink_near(ink, rule, False, band) >= need:
            return rule
    return None


def _remove_rules_and_labels(ink, rules, label_margin):
    h, w = ink.shape
    cleaned = ink.copy()
    # Near each rule, erase only pixels that are part of a long horizontal run: the rule itself,
    # even where it wobbles, but not the upright parts of letters sitting on it.
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(21, w // 30), 1)))
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
    band = np.zeros_like(ink)
    reach = max(12, h // 45)
    for y0, y1, x0, x1, _wide in rules:
        cv2.line(band, (int(x0) - 12, int(y0)), (int(x1) + 12, int(y1)), 255, 2 * reach)
    cleaned[(horizontal > 0) & (band > 0)] = 0

    if rules:
        if len(rules) >= 3:
            # Ruled paper: nothing is written under the last line (paper edge, shadow, desk).
            last = rules[-1]
            cleaned[int(min(h, max(last[0], last[1]) + reach)):, :] = 0
        margin = min(r[2] for r in rules)
        top = 0.0
        for y0, y1, x0, x1, _wide in rules:
            if x0 - margin > label_margin * w:
                # Printed label before the underline ("Name:"): keep only what is written on the line.
                band_top = int(max(0, top))
                band_bottom = int(min(h, max(y0, y1) + 2))
                cleaned[band_top:band_bottom, :max(0, int(x0) - 8)] = 0
            top = max(y0, y1) + 2
    return cleaned


# ---------------------------------------------------------------------- thinning

def _thin(binary):
    """Zhang-Suen thinning on a boolean image. Returns a boolean skeleton."""
    img = np.pad(binary.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]; p3 = img[:-2, 2:]; p4 = img[1:-1, 2:]; p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]; p7 = img[2:, :-2]; p8 = img[1:-1, :-2]; p9 = img[:-2, :-2]
            core = img[1:-1, 1:-1]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            remove = (core == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & c1 & c2
            if remove.any():
                core[remove] = 0
                changed = True
    return _clean_corners(img[1:-1, 1:-1].astype(bool))


# Ring order around a pixel: N, NE, E, SE, S, SW, W, NW.
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _clean_corners(skeleton):
    """Zhang-Suen leaves staircase pixels (an L where a diagonal step would do). Each one looks like a
    junction to the tracer and chops curves into tiny pieces, which is what made the neon dotted.
    A pixel whose neighbours form ONE connected run can go without breaking the line (a simple point)."""
    pixels = set(zip(*np.nonzero(skeleton)))
    changed = True
    while changed:
        changed = False
        for p in sorted(pixels):
            y, x = p
            ring = [((y + dy, x + dx) in pixels) for dy, dx in _RING]
            count = sum(ring)
            if count < 2:
                continue
            runs = sum(1 for i in range(8) if ring[i] and not ring[i - 1])
            if runs == 1:
                pixels.discard(p)
                changed = True
    out = np.zeros_like(skeleton)
    for y, x in pixels:
        out[y, x] = True
    return out


# ---------------------------------------------------------------------- tracing

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace(skeleton, w):
    """Walks the skeleton into polylines.

    A pixel's role comes from the number of separate runs of neighbours around it (not the neighbour
    count): 1 run is a line end, 2 runs is the middle of a line, 3+ is a real junction. Counting
    neighbours mistook every staircase corner for a junction and chopped curves into dashes.
    """
    pixels = set(zip(*np.nonzero(skeleton)))

    def ring_of(p):
        y, x = p
        return [((y + dy, x + dx) if (y + dy, x + dx) in pixels else None) for dy, dx in _RING]

    def runs_of(ring):
        """Neighbour pixels grouped into cyclic runs, each run a list of pixels."""
        present = [r is not None for r in ring]
        if all(present):
            return [list(ring)]
        runs, current = [], []
        start = next(i for i in range(8) if not present[i])      # begin at a gap so no run wraps around
        for k in range(1, 9):
            i = (start + k) % 8
            if present[i]:
                current.append(ring[i])
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    info = {p: runs_of(ring_of(p)) for p in pixels}
    kind = {p: len(r) for p, r in info.items()}          # 0 = isolated, 1 = end, 2 = line, 3+ = junction
    visited = set()
    paths = []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def pick(run):
        """From a run of neighbours, the best next pixel (prefer a straight 4-neighbour)."""
        return min(run, key=lambda q: 0 if q is run[len(run) // 2] else 1)

    def walk(start, nxt):
        path = [start, nxt]
        visited.add(edge(start, nxt))
        prev, cur = start, nxt
        while kind.get(cur, 0) == 2:
            runs = info[cur]
            came = next((i for i, r in enumerate(runs) if prev in r), None)
            if came is None:
                break
            other = runs[1 - came]
            for q in runs[came]:                  # pixels beside where we came from are redundant
                visited.add(edge(cur, q))
            n = pick(other)
            if edge(cur, n) in visited:
                break
            for q in other:
                visited.add(edge(cur, q))
            path.append(n)
            prev, cur = cur, n
        return path

    starts = [p for p in pixels if kind[p] == 1] + [p for p in pixels if kind[p] >= 3]
    for s_ in starts:
        for run in info[s_]:
            for n in run:
                if edge(s_, n) not in visited:
                    paths.append(walk(s_, n))
    for p in pixels:       # closed loops (an "o" has no end) are what is left
        for run in info[p]:
            for n in run:
                if edge(p, n) not in visited:
                    paths.append(walk(p, n))

    min_len = MIN_STROKE_FRAC * w
    spur = SPUR_FRAC * w
    result = []
    for path in paths:
        if len(path) < 3:
            continue
        a, b = path[0], path[-1]
        # A short side branch (one free end, the other end on a junction) is thinning noise, not writing.
        if len(path) < spur and ((kind.get(a, 0) == 1) != (kind.get(b, 0) == 1)):
            continue
        pts = np.array([[x, y] for y, x in path], dtype=np.float32).reshape(-1, 1, 2)
        simple = cv2.approxPolyDP(pts, 1.2, False).reshape(-1, 2)
        if len(simple) < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(simple, axis=0), axis=1)))
        if length < min_len:
            continue
        result.append([(float(x), float(y)) for x, y in simple])
    return _join(result, JOIN_FRAC * w)


def _direction(points, from_end, look=6):
    """Unit direction a stroke is heading as it leaves the given end (pointing away from the stroke)."""
    pts = points[-look:][::-1] if from_end else points[:look]
    if len(pts) < 2:
        return (0.0, 0.0)
    dx, dy = pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)


def _join(strokes, gap):
    """Joins pieces whose ends meet, choosing the continuation that keeps the pen heading the same way
    (so a letter is one flowing line, not a zigzag through a junction)."""
    strokes = [list(s) for s in strokes]
    alive = [True] * len(strokes)
    changed = True
    while changed:
        changed = False
        best = None      # (score, i, j, flip)
        for i in range(len(strokes)):
            if not alive[i]:
                continue
            tail = strokes[i][-1]
            d1 = _direction(strokes[i], True)
            for j in range(len(strokes)):
                if j == i or not alive[j]:
                    continue
                for flip in (False, True):
                    cand = strokes[j][::-1] if flip else strokes[j]
                    gap_now = _dist(tail, cand[0])
                    if gap_now > gap:
                        continue
                    # direction of the candidate as it leaves the join point
                    d2 = _direction(cand, False)
                    dot = d1[0] * d2[0] + d1[1] * d2[1]
                    if dot < 0.25:        # sharper than ~75 degrees: a different stroke, not a continuation
                        continue
                    score = dot - gap_now / (gap * 4.0)
                    if best is None or score > best[0]:
                        best = (score, i, j, flip)
        if best is not None:
            _, i, j, flip = best
            other = strokes[j][::-1] if flip else strokes[j]
            strokes[i] = strokes[i] + other[1:]
            alive[j] = False
            changed = True
    result = [s for i, s in enumerate(strokes) if alive[i]]
    # A letter like "o" or "a" whose pen line almost meets itself: close the loop.
    for s in result:
        if len(s) > 12 and gap * 0.3 < _dist(s[0], s[-1]) <= gap * 1.8:
            s.append(s[0])
    return result


def _smooth(stroke, w):
    """One round of Chaikin corner cutting (stay true to the cleaned image), then even resampling."""
    pts = np.array(stroke, dtype=np.float32)
    for _ in range(1):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mixed = np.empty((len(q) * 2, 2), np.float32)
        mixed[0::2] = q
        mixed[1::2] = r
        pts = np.vstack([pts[:1], mixed, pts[-1:]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] < 1e-3:
        return [(float(pts[0][0]), float(pts[0][1]))] * 2
    count = max(2, int(dist[-1] / (SPACING_FRAC * w)) + 1)
    t = np.linspace(0.0, dist[-1], count)
    return list(zip(np.interp(t, dist, pts[:, 0]).tolist(), np.interp(t, dist, pts[:, 1]).tolist()))


def _keep(stroke, w, h):
    """Drops specks and anything hugging the crop border (paper edge, shadow, desk)."""
    length = sum(_dist(stroke[i], stroke[i + 1]) for i in range(len(stroke) - 1))
    if length < MIN_STROKE_FRAC * w:
        return False
    xs = [p[0] for p in stroke]
    ys = [p[1] for p in stroke]
    # A long, straight, level fragment is the remains of a ruled line (hand-drawn rules hook at the ends).
    if max(xs) - min(xs) >= 0.15 * w and max(ys) - min(ys) <= 0.012 * w:
        return False
    # A hand-drawn divider wobbles more: any long, flat stroke is a line, not writing.
    sw, sh = max(xs) - min(xs), max(ys) - min(ys)
    if sw >= 0.22 * w and sh <= 0.045 * w and sw >= 7.0 * max(sh, 1.0):
        return False
    # Likewise a tall, straight, upright fragment is the paper's edge, not handwriting.
    if max(ys) - min(ys) >= 0.14 * w and max(xs) - min(xs) <= 0.012 * w:
        return False
    # ...and a tall thin one hugging the left or right border is an edge shadow or the desk.
    band = 0.10 * w
    cx = sum(xs) / len(xs)
    if (cx < band or cx > w - band) and max(ys) - min(ys) >= 0.06 * w and (max(ys) - min(ys)) > 3.0 * max(1.0, max(xs) - min(xs)):
        return False
    edge = 0.04 * w
    if min(xs) < edge and max(xs) < 2 * edge or max(xs) > w - edge and min(xs) > w - 2 * edge:
        return False
    if min(ys) < edge and max(ys) < 2 * edge or max(ys) > h - edge and min(ys) > h - 2 * edge:
        return False
    return True


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------- name / message, order

def _split(strokes, divider):
    """Strokes above the divider are the name, below are the message. No divider: all message."""
    if divider is None:
        return [], strokes
    y0, y1, x0, x1 = divider[:4]
    name, message = [], []
    for s in strokes:
        cx = sum(p[0] for p in s) / len(s)
        cy = sum(p[1] for p in s) / len(s)
        t = 0.0 if x1 == x0 else min(1.0, max(0.0, (cx - x0) / (x1 - x0)))
        line_y = y0 + (y1 - y0) * t
        (name if cy < line_y else message).append(s)
    if not name or not message:
        return [], strokes          # writing on one side only: treat it all as the message
    return name, message


def _order(strokes, height):
    """Writing order: group into text rows by vertical centre, then left to right within a row."""
    if not strokes:
        return strokes
    info = []
    for s in strokes:
        ys = [p[1] for p in s]
        xs = [p[0] for p in s]
        info.append((sum(ys) / len(ys), min(xs), s))
    info.sort(key=lambda t: t[0])
    row_gap = max(14.0, height * 0.045)
    rows, current, last_y = [], [], None
    for cy, minx, s in info:
        if last_y is not None and cy - last_y > row_gap:
            rows.append(current)
            current = []
        current.append((minx, s))
        last_y = cy if last_y is None else (last_y * 0.7 + cy * 0.3)
    rows.append(current)
    ordered = []
    for row in rows:
        for _, s in sorted(row, key=lambda t: t[0]):
            ordered.append(s if s[0][0] <= s[-1][0] else s[::-1])    # a pen moves left to right
    return ordered


# ---------------------------------------------------------------------- debug preview

def render_preview(result, width=900):
    """What the wall will draw: the strokes in neon yellow (name in cyan) on black, with the divider."""
    aspect = result["aspect"]
    height = int(width * aspect)
    canvas = np.zeros((height, width, 3), np.uint8)
    if result.get("divider") is not None:
        y = int(result["divider"] * width)
        for x in range(0, width, 24):
            cv2.line(canvas, (x, y), (min(width, x + 12), y), (90, 90, 90), 1)
    for s in result["strokes"]:
        pts = (np.array(s["p"], dtype=np.float32).reshape(-1, 2) * width).astype(np.int32)
        colour = (255, 230, 90) if s.get("g", 0) == 1 else (50, 220, 255)     # BGR: name = cyan, message = yellow
        glow = canvas.copy()
        cv2.polylines(glow, [pts], False, colour, 5, cv2.LINE_AA)
        canvas = cv2.addWeighted(glow, 0.35, canvas, 0.65, 0)
        cv2.polylines(canvas, [pts], False, (230, 250, 255), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], False, colour, 2, cv2.LINE_AA)
    return canvas


def render_overlay(result, clean, width=900):
    """The cleaned image with the traced strokes drawn over it, so every mark can be checked against it."""
    h, w = clean.shape[:2]
    base = cv2.resize(clean, (width, int(width * h / float(w))), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor((base * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for s in result["strokes"]:
        pts = (np.array(s["p"], dtype=np.float32).reshape(-1, 2) * width).astype(np.int32)
        colour = (255, 230, 90) if s.get("g", 0) == 1 else (50, 220, 255)
        cv2.polylines(canvas, [pts], False, colour, 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------- is it the same note again?

FP_W, FP_H = 160, 120


def fingerprint(clean):
    """A small picture of the note's ink, independent of where on the desk the note lies: the ink is cropped to
    its own bounding box and shrunk to a fixed size. Used only to tell whether a note was already sent."""
    ys, xs = np.nonzero(clean > 0)
    if len(xs) < 20:
        return None
    crop = (clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8) * 255
    small = cv2.resize(crop, (FP_W, FP_H), interpolation=cv2.INTER_AREA)
    return (small > 40).astype(np.uint8)


def _band_score(a, b, k, bands):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    da, db = cv2.dilate(a, kernel), cv2.dilate(b, kernel)
    worst = 1.0
    h = a.shape[0]
    for i in range(bands):
        s = slice(i * h // bands, (i + 1) * h // bands)
        sa, sb = float(a[s].sum()), float(b[s].sum())
        if sa < 15 and sb < 15:
            continue            # nothing written in this band on either note
        ra = float((a[s] & db[s]).sum()) / max(1.0, sa)
        rb = float((b[s] & da[s]).sum()) / max(1.0, sb)
        worst = min(worst, ra, rb)
    return worst


def same_note(a, b, threshold=0.92):
    """True when two fingerprints show the same writing. The note is compared in horizontal bands and the WORST band
    decides, so two notes that share a name but differ in the deed are not mistaken for each other, while a note that
    was lifted and put back (moved a little, slightly tilted, pen a little thicker) still matches."""
    if a is None or b is None:
        return False
    best = 0.0
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            best = max(best, _band_score(a, shifted, 5, 4))
    return best >= threshold
