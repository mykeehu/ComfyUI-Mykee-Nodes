"""
ComfyUI-Mykee-Nodes / face refine nodes

A model-agnostic face "detailer" split into small, independent pieces, so it
works with ANY image or video model (and with latents VAE-Encode/Decode
cannot handle directly, such as MiniMax H3's packed video+audio latent):

  1) MykeeFaceTracks   - finds every face in an IMAGE batch (a single image is
                         a 1-frame video), links them into per-person tracks
                         over time, decides WHO each track is (SFace identity
                         match against your reference images), and cuts out
                         one square, eye-levelled crop per face and frame.
                         Per person it also outputs a clean reference face
                         (from your source image) and the best face found in
                         the image itself (self reference).
  2) (your own sampler chain - any model, any conditioning)
  3) MykeeH3FaceLatent - H3 only: turns the crops into H3 video+audio latents.
  4) MykeeFacePaste    - blends the refined crops back into the original
                         frames with a soft face mask, optional colour match
                         and an identity guard (a refined face is only used
                         when it is not LESS similar to the reference than
                         the original face was).

The geometry (similarity transform original -> crop) is stored in the
FACE_TRACKS object, so partial faces at the image border, tilted heads and
moving faces in videos all paste back to the exact place they came from.
"""

import math
import os

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # scipy is optional; a greedy fallback is used instead
    _linear_sum_assignment = None

from .mykee_character import (
    cv2,
    _require_cv2,
    _resolve_model_path,
    _list_annotator_files,
    _default_annotator_choice,
    _edge_fill_cv2_params,
    _embed_face_sface,
    _cosine_similarity,
    _toast,
    _select_face,
)

try:
    import comfy.utils as _comfy_utils
except Exception:  # running outside ComfyUI (tests)
    _comfy_utils = None

MAX_PEOPLE = 10
CATEGORY = "Mykee/Face Refine"
FRAME_PICKS = ["middle", "first", "last"]

# A track's identity is only called "ambiguous" when its best and second-best
# person scores are closer than this.
AMBIGUITY_MARGIN = 0.05
_ASSIGN_MODES = ("by_similarity", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _first(x):
    """INPUT_IS_LIST nodes receive every input as a list; unwrap one."""
    return x[0] if isinstance(x, (list, tuple)) else x


def _progress(total):
    if _comfy_utils is not None and total > 1:
        try:
            return _comfy_utils.ProgressBar(int(total))
        except Exception:
            return None
    return None


def _frame_bgr(images, t):
    """IMAGE batch frame t (H,W,C float 0..1 RGB) -> uint8 BGR."""
    arr = images[t].detach().cpu().numpy()[..., :3]
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)


def _bgr_to_rgb_f32(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _rgb_f32_to_bgr_u8(rgb):
    arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)


def _gray_image(size):
    return torch.full((1, size, size, 3), 0.5, dtype=torch.float32)


def _scale_of(matrix):
    return float(math.sqrt(abs(np.linalg.det(matrix[:, :2]))))


# ---------------------------------------------------------------------------
# face detection (YuNet), with a cached detector instance
# ---------------------------------------------------------------------------

_DETECTOR_CACHE = {}


def _get_detector(path, score_threshold):
    """One YuNet instance per model file, reused for every frame (creating a
    fresh cv2.dnn net per frame is slow and has tripped an OpenCV
    shape-inference bug before - see mykee_character.py)."""
    det = _DETECTOR_CACHE.get(path)
    if det is None:
        det = cv2.FaceDetectorYN_create(path, "", (320, 320), float(score_threshold), 0.3, 5000)
        _DETECTOR_CACHE[path] = det
    else:
        det.setScoreThreshold(float(score_threshold))
    return det


def _detect_rows(bgr, yunet_path, score_threshold, min_face_px=0, upscale=1.0):
    """YuNet rows (N,15): x,y,w,h, 5 landmarks (x,y), score - in the
    coordinates of `bgr`. `upscale` > 1 runs the detector on an enlarged copy
    (helps with tiny faces); the rows are mapped back to bgr's coordinates."""
    h, w = bgr.shape[:2]
    work = bgr
    if abs(upscale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if upscale > 1.0 else cv2.INTER_AREA
        work = cv2.resize(
            bgr, (max(8, int(round(w * upscale))), max(8, int(round(h * upscale)))), interpolation=interp
        )
    det = _get_detector(yunet_path, score_threshold)
    det.setInputSize((work.shape[1], work.shape[0]))
    _, faces = det.detect(work)
    if faces is None or len(faces) == 0:
        return np.zeros((0, 15), np.float64)
    faces = np.asarray(faces, dtype=np.float64)
    faces = faces[np.isfinite(faces).all(axis=1)]
    if abs(upscale - 1.0) > 1e-3:
        faces[:, :14] /= upscale
    if len(faces):
        faces = faces[(faces[:, 2] > 1.0) & (faces[:, 3] > 1.0)]
    if min_face_px > 0 and len(faces):
        faces = faces[np.maximum(faces[:, 2], faces[:, 3]) >= float(min_face_px)]
    return _dedupe_rows(faces)


def _pick_source_face(rows, which, index, image_size):
    """Picks the reference face; 'most_frontal' is handled by the shared
    _select_face (same logic as Mykee Face Detect)."""
    return _select_face(rows, which, index, image_size=image_size)


def _dedupe_rows(faces):
    """YuNet's own NMS lets a second, partly overlapping box of the SAME face
    through now and then (e.g. one box on the face plus one that also takes in
    forehead / hair). Such a duplicate would become a second track and a second
    crop. A lower-scoring box is dropped when (a) most of its smaller side lies
    inside a higher-scoring one, or (b) its own eye / nose landmarks fall inside
    a higher-scoring box - two different faces never have their eyes inside each
    other's box."""
    if len(faces) < 2:
        return faces
    order = np.argsort(-faces[:, 14])
    keep = []

    def inside(b, px, py):
        return b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]

    for i in order:
        a = faces[i]
        dup = False
        for k in keep:
            b = faces[k]
            ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            inter = ix * iy
            if inter / max(1e-9, min(a[2] * a[3], b[2] * b[3])) > 0.6:
                dup = True
                break
            if inter > 0.0:
                eye_mid = ((a[4] + a[6]) * 0.5, (a[5] + a[7]) * 0.5)
                if inside(b, a[8], a[9]) or inside(b, eye_mid[0], eye_mid[1]):
                    dup = True
                    break
        if not dup:
            keep.append(i)
    return faces[sorted(keep)]


# ---------------------------------------------------------------------------
# geometry: face -> crop window -> similarity transform (original -> crop)
# ---------------------------------------------------------------------------

def _face_window(face, crop_factor, align):
    """Crop window for one YuNet row: centre, side (in original pixels) and
    the rotation that levels the eyes, plus a few quality hints.

    'landmarks' framing (steady, FFHQ-like) is used whenever the 5 landmarks
    look sane; otherwise (profile views, tiny or broken landmarks) it falls
    back to the detector's bbox. crop_factor=2.0 is the FFHQ window."""
    x, y, fw, fh = [float(v) for v in face[0:4]]
    re_ = np.array([face[4], face[5]], dtype=np.float64)
    le_ = np.array([face[6], face[7]], dtype=np.float64)
    nose = np.array([face[8], face[9]], dtype=np.float64)
    mouth_mid = np.array([(face[10] + face[12]) * 0.5, (face[11] + face[13]) * 0.5], dtype=np.float64)
    eye_mid = (re_ + le_) * 0.5

    v = le_ - re_
    m = mouth_mid - eye_mid
    e2e = float(np.hypot(*v))
    e2m = float(np.hypot(*m))

    lm_ok = False
    roll = 0.0
    frontality = 0.3
    if e2e > 2.0 and e2m > 2.0:
        u = v / e2e
        down = np.array([-u[1], u[0]])
        ratio = e2m / e2e
        if float(np.dot(m, down)) > 0.0 and 0.5 <= ratio <= 3.0:
            lm_ok = True
            roll = float(np.degrees(np.arctan2(v[1], v[0])))
            nose_off = float(np.dot(nose - eye_mid, u)) / e2e
            frontality = float(np.clip(1.0 - abs(nose_off) / 0.6, 0.0, 1.0))

    if lm_ok:
        c = eye_mid + m * 0.1
        base = max(e2e * 2.0, e2m * 1.8, 0.8 * max(fw, fh))
        side = base * crop_factor
        mode = "landmarks"
    else:
        c = np.array([x + fw / 2.0, y + fh / 2.0])
        side = max(fw, fh) * crop_factor * 0.9
        mode = "bbox"

    angle = roll if (align == "level_eyes" and lm_ok) else 0.0
    return {
        "cx": float(c[0]), "cy": float(c[1]), "side": float(max(side, 4.0)),
        "angle": float(angle), "lm_ok": lm_ok, "roll": roll,
        "frontality": frontality, "mode": mode,
    }


def _window_matrix(cx, cy, side, angle, size):
    """2x3 similarity transform original -> crop (rotate by `angle` around the
    window centre, scale side->size, centre in the crop). Same convention as
    MykeeFaceDetectAlignCrop's level_eyes crop."""
    scale = float(size) / float(side)
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), scale)
    m[0, 2] += size / 2.0 - cx
    m[1, 2] += size / 2.0 - cy
    return m


def _apply(matrix, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def _face_geom_crop(face, matrix, lm_ok=True):
    """Face box + head roll expressed in CROP space (for the soft mask).
    Without usable landmarks (profile views) the mask stays upright."""
    x, y, fw, fh = [float(v) for v in face[0:4]]
    s = _scale_of(matrix)
    c = _apply(matrix, [[x + fw / 2.0, y + fh / 2.0]])[0]
    roll = 0.0
    if lm_ok:
        eyes = _apply(matrix, [[face[4], face[5]], [face[6], face[7]]])
        d = eyes[1] - eyes[0]
        if np.hypot(*d) > 1.0:
            roll = float(np.degrees(np.arctan2(d[1], d[0])))
        if abs(roll) > 90.0:  # landmarks swapped / upside-down: keep the mask upright
            roll = 0.0
    return {"c": (float(c[0]), float(c[1])), "w": fw * s, "h": fh * s, "roll": roll}


def _ellipse_mask(size, geom, grow_pct, forehead_pct, feather_pct):
    """Soft elliptical face mask in crop space. YuNet's box runs roughly from
    the eyebrows to the chin, so the ellipse is extended upwards by
    forehead_pct of the box height. feather_pct is the blur, as a percentage
    of the crop size (resolution independent)."""
    grow = max(0.05, 1.0 + grow_pct / 100.0)
    hw = geom["w"] * grow / 2.0
    hh = geom["h"] * grow / 2.0
    ext = geom["h"] * forehead_pct / 100.0
    th = math.radians(geom["roll"])
    up = np.array([math.sin(th), -math.cos(th)])
    cx = geom["c"][0] + up[0] * ext / 2.0
    cy = geom["c"][1] + up[1] * ext / 2.0
    axes = (max(1, int(round(hw))), max(1, int(round(hh + ext / 2.0))))
    m8 = np.zeros((size, size), np.uint8)  # LINE_AA only works on 8-bit images
    cv2.ellipse(m8, (int(round(cx)), int(round(cy))), axes, float(geom["roll"]), 0, 360, 255, -1, cv2.LINE_AA)
    m = m8.astype(np.float32) / 255.0
    sigma = size * feather_pct / 100.0 / 2.0
    if sigma > 0.3:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def _window_roi(matrix, size, width, height, pad=4):
    """Bounding box (x0,y0,x1,y1), in ORIGINAL pixels, of the crop square."""
    a = matrix[:, :2]
    ainv = np.linalg.inv(a)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    pts = (corners - matrix[:, 2]) @ ainv.T
    x0 = int(math.floor(pts[:, 0].min())) - pad
    y0 = int(math.floor(pts[:, 1].min())) - pad
    x1 = int(math.ceil(pts[:, 0].max())) + pad
    y1 = int(math.ceil(pts[:, 1].max())) + pad
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return x0, y0, x1, y1


def _warp_to_crop(bgr, matrix, size, edge_fill):
    """Cuts the crop out of `bgr` with the stored transform. Works on a small
    ROI (cheap on 4 MP frames) and pre-blurs when the face is larger than the
    crop (warpAffine alone would alias)."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = _window_roi(matrix, size, w, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full((size, size, 3), 127, np.uint8)
    roi = bgr[y0:y1, x0:x1]
    scale = _scale_of(matrix)
    if scale < 0.8:
        roi = cv2.GaussianBlur(roi, (0, 0), 0.45 / scale)
    m2 = matrix.copy()
    m2[:, 2] += matrix[:, :2] @ np.array([x0, y0], dtype=np.float64)
    border_type, border_value = _edge_fill_cv2_params(edge_fill)
    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        roi, m2, (size, size), flags=interp, borderMode=border_type,
        borderValue=border_value if border_value is not None else 0,
    )


def _crop_face(bgr, face, size, crop_factor, align, edge_fill):
    win = _face_window(face, crop_factor, align)
    mat = _window_matrix(win["cx"], win["cy"], win["side"], win["angle"], size)
    return _warp_to_crop(bgr, mat, size, edge_fill), mat


def _row_in_crop(face, matrix):
    """A YuNet row re-expressed in crop coordinates (for embedding a crop)."""
    out = np.array(face, dtype=np.float64).copy()
    s = _scale_of(matrix)
    c = _apply(matrix, [[face[0] + face[2] / 2.0, face[1] + face[3] / 2.0]])[0]
    out[2] = face[2] * s
    out[3] = face[3] * s
    out[0] = c[0] - out[2] / 2.0
    out[1] = c[1] - out[3] / 2.0
    out[4:14] = _apply(matrix, np.array(face[4:14], dtype=np.float64).reshape(5, 2)).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _assignment(cost):
    """Minimum-cost one-to-one assignment: scipy if available, else greedy."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if _linear_sum_assignment is not None:
        r, c = _linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs = []
    used_r, used_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if i in used_r or j in used_c:
            continue
        pairs.append((int(i), int(j)))
        used_r.add(i)
        used_c.add(j)
    return pairs


_BIG = 1e6


def _link_tracks(det, keyframes, max_gap, link_iou, stride):
    """Links per-keyframe detections into tracks (IoU + centre distance,
    one-to-one per frame). A track survives up to `max_gap` frames without a
    detection."""
    limit = max(int(max_gap), int(stride))
    tracks = []
    for f in keyframes:
        rows = det[f]
        active = [i for i, t in enumerate(tracks) if 0 < f - t["last_frame"] <= limit]
        matched = set()
        if active and len(rows):
            cost = np.full((len(active), len(rows)), _BIG)
            for ai, ti in enumerate(active):
                tr = tracks[ti]
                gap = f - tr["last_frame"]
                lb = tr["last_box"]
                # where the face should be now: last position + constant velocity.
                # This is what keeps two faces that cross paths on their own tracks.
                pc = np.array([lb[0] + lb[2] / 2.0, lb[1] + lb[3] / 2.0]) + tr["vel"] * gap
                pb = np.array([pc[0] - lb[2] / 2.0, pc[1] - lb[3] / 2.0, lb[2], lb[3]])
                for j in range(len(rows)):
                    r = rows[j]
                    iou = _iou(pb, r[:4])
                    rc = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0])
                    d = float(np.hypot(*(rc - pc))) / max(lb[2], lb[3], r[2], r[3], 1.0)
                    gate = min(2.0, 0.6 * (1.0 + 0.15 * (gap - 1)))
                    if iou >= link_iou or d <= gate:
                        cost[ai, j] = (1.0 - iou) + 0.5 * d
            for ai, j in _assignment(cost):
                if cost[ai, j] >= _BIG * 0.5:
                    continue
                tr = tracks[active[ai]]
                _extend_track(tr, f, rows[j])
                matched.add(j)
        for j in range(len(rows)):
            if j not in matched:
                tracks.append({
                    "kf": [(f, rows[j])], "last_frame": f, "last_box": rows[j][:4].copy(),
                    "vel": np.zeros(2),
                })
    return tracks


def _extend_track(tr, f, row):
    """Adds a detection to a track and updates its velocity estimate
    (pixels / frame, lightly smoothed)."""
    lb = tr["last_box"]
    gap = max(1, f - tr["last_frame"])
    old_c = np.array([lb[0] + lb[2] / 2.0, lb[1] + lb[3] / 2.0])
    new_c = np.array([row[0] + row[2] / 2.0, row[1] + row[3] / 2.0])
    v = (new_c - old_c) / gap
    tr["vel"] = v if len(tr["kf"]) == 1 else 0.6 * v + 0.4 * tr["vel"]
    tr["kf"].append((f, row))
    tr["last_frame"] = f
    tr["last_box"] = row[:4].copy()


def _interpolate_track(kf):
    """Fills the frames between a track's detections by linear interpolation
    of the whole YuNet row. Returns (frames, rows, is_detected)."""
    fs = np.array([k[0] for k in kf], dtype=np.float64)
    rs = np.stack([k[1] for k in kf]).astype(np.float64)
    frames = np.arange(int(fs[0]), int(fs[-1]) + 1)
    rows = np.zeros((len(frames), rs.shape[1]), np.float64)
    for col in range(rs.shape[1]):
        rows[:, col] = np.interp(frames.astype(np.float64), fs, rs[:, col])
    detected = np.isin(frames, fs.astype(int))
    return frames.tolist(), rows, detected.tolist()


def _smooth_windows(wins, mode, radius):
    """Temporal smoothing of one track's crop windows. Centre and angle get a
    gaussian moving average; the window size is either one fixed value per
    track (median - the best for video: constant scale, no zoom flicker) or
    smoothed per frame."""
    n = len(wins)
    cx = np.array([w["cx"] for w in wins])
    cy = np.array([w["cy"] for w in wins])
    side = np.array([w["side"] for w in wins])
    ang = np.array([w["angle"] for w in wins])
    ok = np.array([w["lm_ok"] for w in wins])

    if n > 1:
        # frames without usable landmarks have angle 0 - borrow the nearest
        # valid angle so the average does not jump
        if ok.any() and not ok.all():
            idx = np.arange(n)
            ang = np.interp(idx, idx[ok], ang[ok])
        ang = np.degrees(np.unwrap(np.radians(ang)))

        def gauss(arr):
            r = max(0, int(radius))
            if r == 0:
                return arr
            sigma = max(0.5, r / 2.0)
            out = np.empty_like(arr)
            for i in range(n):
                lo, hi = max(0, i - r), min(n, i + r + 1)
                k = np.arange(lo, hi) - i
                wgt = np.exp(-(k ** 2) / (2.0 * sigma ** 2))
                out[i] = float(np.sum(arr[lo:hi] * wgt) / np.sum(wgt))
            return out

        cx, cy, ang = gauss(cx), gauss(cy), gauss(ang)
        side = np.full(n, float(np.median(side))) if mode == "per_track_median" else gauss(side)

    out = []
    for i, w in enumerate(wins):
        nw = dict(w)
        nw.update(cx=float(cx[i]), cy=float(cy[i]), side=float(side[i]), angle=float(ang[i]))
        out.append(nw)
    return out


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def _safe_embed(bgr, face, sface_path):
    try:
        return _embed_face_sface(bgr, face, sface_path)
    except Exception:
        return None


def _mean_embedding(embs):
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    arr = np.stack([e / (np.linalg.norm(e) + 1e-8) for e in embs])
    mean = arr.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


def _assign_tracks(track_embs, person_embs, threshold, exclusive):
    """Track -> person. Default: every track goes to its best-matching person
    (so one person can own many tracks, e.g. several panels of a character
    sheet), but only above `threshold`. `exclusive` makes it one track per
    person (Hungarian). Returns (assigned, sims, ambiguous)."""
    persons = sorted(person_embs)
    n_t = len(track_embs)
    sims = np.full((n_t, len(persons)), -1.0)
    for i, e in enumerate(track_embs):
        if e is None:
            continue
        for j, p in enumerate(persons):
            sims[i, j] = _cosine_similarity(e, person_embs[p])

    assigned = [None] * n_t
    ambiguous = [False] * n_t
    if not persons or n_t == 0:
        return assigned, sims, ambiguous

    if exclusive:
        for i, j in _assignment(-sims):
            if sims[i, j] >= threshold:
                assigned[i] = persons[j]
    else:
        for i in range(n_t):
            j = int(np.argmax(sims[i]))
            if sims[i, j] >= threshold:
                assigned[i] = persons[j]
                if len(persons) > 1:
                    second = float(np.sort(sims[i])[-2])
                    ambiguous[i] = second >= threshold and (sims[i, j] - second) < AMBIGUITY_MARGIN
    return assigned, sims, ambiguous


# ---------------------------------------------------------------------------
# Node 1: Mykee Face Tracks
# ---------------------------------------------------------------------------

class MykeeFaceTracks:
    """
    Detect + track + identify + crop, for images and videos alike.

    A single IMAGE is treated as a 1-frame video, so there is one code path.
    Every face becomes one square crop (`crop_size`), levelled on the eyes,
    cut with a similarity transform that is stored in the FACE_TRACKS output;
    MykeeFacePaste uses it to put the refined crop back exactly where it came
    from (also for faces cut off by the image border).

    Identity: each track (the same face followed through the frames) gets one
    SFace embedding, the mean over its best frames, and is matched to the
    person whose `source_image_N` it resembles most - decided per TRACK, so a
    person cannot flip between frames, and faces cannot be swapped between
    persons. One person may own many tracks (several faces of the same
    character on one sheet).

    Outputs (lists, aligned entry by entry, person 1 first):
      crops       all crops of all people - run them through your own chain.
      masks       face-shaped soft masks in crop space.
      ref_faces   the face of the person's source_image_N, cut with the SAME
                  framing as the crops - feed it to your model as the identity
                  reference (falls back to self_faces if no source image).
      self_faces  the best face found IN the input (large, frontal, sharp
                  detection) - e.g. the close-up panel of a character sheet,
                  to use as a second reference for the small faces.
      person      the person number of each entry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        yunet_files = _list_annotator_files(".onnx") or ["face_detection_yunet_2023mar.onnx"]
        sface_files = _list_annotator_files(".onnx") or ["face_recognition_sface_2021dec.onnx"]
        required = {
            "images": ("IMAGE", {"tooltip": "One image, or a video as a frame batch."}),
            "num_people": ("INT", {
                "default": 1, "min": 1, "max": MAX_PEOPLE, "step": 1,
                "tooltip": "How many different people to look for (1-10). Adds/removes the source_image_N inputs and the crops_N / masks_N / ref_face_N / self_face_N outputs.",
            }),
            "model_file_yunet": (yunet_files, {"default": _default_annotator_choice(yunet_files, "face_detection_yunet_2023mar.onnx")}),
            "model_file_sface": (sface_files, {"default": _default_annotator_choice(sface_files, "face_recognition_sface_2021dec.onnx")}),
            "crop_size": ("INT", {
                "default": 768, "min": 128, "max": 2048, "step": 16,
                "tooltip": "Side of every output crop. Use a size your model likes; for MiniMax H3 it must be a multiple of 16 (768 or 1024 are good).",
            }),
            "crop_factor": ("FLOAT", {
                "default": 2.0, "min": 0.8, "max": 6.0, "step": 0.1,
                "tooltip": "How much surrounding context each crop contains. 2.0 = the FFHQ head window (face + hair + a bit of neck); 1.4 = tighter; 3.0 = lots of context.",
            }),
            "align": (["level_eyes", "none"], {
                "default": "level_eyes",
                "tooltip": "level_eyes rotates every crop so the eyes are horizontal (the model sees upright faces; the paste node rotates it back). Profile views / unreliable landmarks are never rotated.",
            }),
            "score_threshold": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01}),
            "min_face_size_px": ("INT", {
                "default": 24, "min": 0, "max": 4096, "step": 1,
                "tooltip": "Ignore detections whose larger side is below this many pixels (in the input resolution). Absolute on purpose: a small full-body face must NOT be dropped just because another panel has a big close-up. 0 = off.",
            }),
            "detect_upscale": ("FLOAT", {
                "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.25,
                "tooltip": "Run the detector on an enlarged copy of each frame. Helps with very small faces; costs time.",
            }),
            "match_threshold": ("FLOAT", {
                "default": 0.25, "min": -1.0, "max": 1.0, "step": 0.01,
                "tooltip": "Minimum SFace similarity for a track to count as a given person. Faces below it are left alone (listed as unmatched in the summary). Lower = more permissive.",
            }),
            "exclusive_matching": ("BOOLEAN", {
                "default": False,
                "tooltip": "Each person may own at most ONE track. Only use this when every person appears exactly once (it stops two similar-looking people from claiming the same face). Leave off for character sheets where one person appears several times.",
            }),
            "skip_self_reference": ("BOOLEAN", {
                "default": True,
                "tooltip": "Still images only: when a person has several faces, the best one (used as self_face_N) is not included in crops_N - it is your reference, you do not want to redraw it. Ignored for videos.",
            }),
            "mask_grow_pct": ("FLOAT", {"default": 10.0, "min": -50.0, "max": 200.0, "step": 1.0, "tooltip": "Grows (+) or shrinks (-) the face mask relative to the detected face box."}),
            "mask_forehead_pct": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Extends the mask upwards (YuNet's box starts at the eyebrows) - percent of the face box height."}),
            "mask_feather_pct": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Edge softness as a percentage of the crop size."}),
            "edge_fill": (["replicate", "reflect", "constant_gray", "constant_black"], {
                "default": "replicate",
                "tooltip": "What fills the crop where the window reaches past the image border (partial faces).",
            }),
            "detect_stride": ("INT", {
                "default": 1, "min": 1, "max": 30, "step": 1,
                "tooltip": "Video: detect on every Nth frame only and interpolate in between. 1 = every frame (most robust).",
            }),
            "max_gap": ("INT", {
                "default": 12, "min": 1, "max": 120, "step": 1,
                "tooltip": "Video: a face that is missed for up to this many frames (turned away, blurred) is still the same track; the gap is filled by interpolation.",
            }),
            "link_iou": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.9, "step": 0.01, "tooltip": "Video: minimum box overlap to link a detection to a track (a centre-distance test also applies, for fast motion)."}),
            "window_size_mode": (["per_track_median", "per_frame"], {
                "default": "per_track_median",
                "tooltip": "Video: per_track_median keeps one constant crop scale per track (no zoom flicker). per_frame lets the window follow the face size.",
            }),
            "smooth_radius": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1, "tooltip": "Video: temporal smoothing of the crop window position/angle (frames each side). 0 = off."}),
            "embed_samples": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "How many of a track's best frames are used for its identity embedding."}),
            "source_which_face": ([
                "largest", "highest_score", "leftmost", "rightmost", "topmost", "bottommost",
                "top_left", "top_right", "bottom_left", "bottom_right", "x_order", "y_order",
                "most_frontal",
            ], {
                "default": "largest",
                "tooltip": "Which face of a source_image_N is the reference person, when the image shows several (e.g. a character sheet). Same options as 'which_face' in Mykee Face Detect, plus most_frontal: the face turned most towards the camera (the largest among the frontal ones) - for character sheets where a profile view is bigger than the frontal close-up. Applies to every source_image_N.",
            }),
            "source_face_index": ("INT", {
                "default": 0, "min": 0, "max": 99, "step": 1,
                "tooltip": "Only used when source_which_face is x_order / y_order: 0-based position in the left-to-right / top-to-bottom ordering.",
            }),
            "chunk_size": ("INT", {
                "default": 1, "min": 1, "max": 256, "step": 1,
                "tooltip": "Crops per list entry (an entry never mixes people). 1 = every crop is processed on its own, which is needed when the crop goes into the conditioning. Larger = faster for a plain VAE Encode -> sampler chain without image conditioning (video).",
            }),
            "assign_mode": (list(_ASSIGN_MODES), {
                "default": "by_similarity",
                "tooltip": "How faces in the input are given to persons. by_similarity: the face that looks most like source_image_N becomes person N (refining / keeping identities). The others hand out the persons by POSITION: the 1st face in that order = person 1 (source_image_1), the 2nd = person 2, ... Use them for FACE SWAPS (the sources are different people, similarity is meaningless) - e.g. left_to_right: source_image_1 goes to the leftmost face. Only the num_people best detections are used (false positives are ignored); match_threshold and exclusive_matching do not apply.",
            }),
        }
        optional = {f"source_image_{i}": ("IMAGE",) for i in range(1, MAX_PEOPLE + 1)}
        return {"required": required, "optional": optional, "hidden": {"node_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING", "FACE_TRACKS", "IMAGE", "MASK", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("summary", "tracks", "crops", "masks", "ref_faces", "self_faces", "person")
    OUTPUT_IS_LIST = (False, False, True, True, True, True, True)
    OUTPUT_TOOLTIPS = (
        "What was found and how it was assigned (per person: tracks, crops, similarity, warnings).",
        "Geometry and identities for Mykee Face Paste.",
        "All crops of ALL people as a list (person 1 first, then 2, ...). The nodes after it run once per entry.",
        "Face-shaped soft masks in crop space, aligned with crops.",
        "Per entry: the reference face (source_image_N) of the person this crop belongs to, cut with the same framing as the crops.",
        "Per entry: the best face found IN the input for that person (e.g. the close-up of a character sheet).",
        "Per entry: the person number (1-based).",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, chunk_size=1, **kwargs):
        """Runs the detection/tracking and turns the per-person results into
        aligned LISTS. ComfyUI runs the nodes after a list once per entry and
        pairs several lists entry by entry, so every crop arrives together with
        ITS OWN person's reference face - one refine chain serves every person."""
        summary, tracks_obj, *outs = self._run_core(**kwargs)
        size = int(tracks_obj["crop_size"])
        step = max(1, int(chunk_size))
        gray = _gray_image(size)
        gray_mask = torch.zeros((1, size, size), dtype=torch.float32)
        crops, masks, refs, selfs, persons = [], [], [], [], []
        for p in sorted(tracks_obj["person_items"]):
            if p > int(tracks_obj["num_people"]) or not tracks_obj["person_items"][p]:
                continue
            base = 4 * (p - 1)
            crops_t, mask_t, ref_t, self_t = outs[base], outs[base + 1], outs[base + 2], outs[base + 3]
            for i in range(0, int(crops_t.shape[0]), step):
                crops.append(crops_t[i:i + step])
                masks.append(mask_t[i:i + step])
                refs.append(ref_t)
                selfs.append(self_t)
                persons.append(int(p))
        if not crops:
            # nothing to refine: one placeholder entry keeps the chain valid; Face Paste ignores it
            crops, masks, refs, selfs, persons = [gray], [gray_mask], [gray], [gray], [0]
        return (summary, tracks_obj, crops, masks, refs, selfs, persons)

    def _run_core(self, images, num_people, model_file_yunet, model_file_sface, crop_size, crop_factor,
            align, score_threshold, min_face_size_px, detect_upscale, match_threshold,
            exclusive_matching, skip_self_reference, mask_grow_pct, mask_forehead_pct,
            mask_feather_pct, edge_fill, detect_stride, max_gap, link_iou, window_size_mode,
            smooth_radius, embed_samples, source_which_face="largest", source_face_index=0,
            assign_mode="by_similarity", node_id=None, **kwargs):
        _require_cv2()
        yunet_path = _resolve_model_path(model_file_yunet)
        sface_path = _resolve_model_path(model_file_sface)
        if not os.path.isfile(yunet_path):
            raise FileNotFoundError(f"YuNet model not found: {yunet_path}.")
        if not os.path.isfile(sface_path):
            raise FileNotFoundError(f"SFace model not found: {sface_path}.")

        size = int(crop_size)
        n_frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        lines = [f"Input: {n_frames} frame(s) of {width}x{height}."]

        # ---- 1) detection on the keyframes ---------------------------------
        stride = max(1, int(detect_stride))
        keyframes = list(range(0, n_frames, stride))
        if keyframes[-1] != n_frames - 1:
            keyframes.append(n_frames - 1)
        pbar = _progress(len(keyframes))
        det = {}
        for f in keyframes:
            rows = _detect_rows(_frame_bgr(images, f), yunet_path, score_threshold, min_face_size_px, detect_upscale)
            if len(rows):
                rows = rows[np.argsort(rows[:, 0] + rows[:, 2] / 2.0)]  # left to right: stable order
            det[f] = rows
            if pbar is not None:
                pbar.update(1)
        total_det = sum(len(v) for v in det.values())
        lines.append(f"{total_det} face detection(s) on {len(keyframes)} frame(s).")
        if n_frames == 1 and total_det:
            for k, r in enumerate(det[keyframes[0]], 1):
                lines.append(
                    f"  face {k}: x={r[0]:.0f} y={r[1]:.0f} {r[2]:.0f}x{r[3]:.0f}px, detector score {r[14]:.2f}"
                )

        # ---- 2) tracks ------------------------------------------------------
        raw_tracks = _link_tracks(det, keyframes, max_gap, link_iou, stride)
        track_list = []
        for tr in raw_tracks:
            frames, rows, is_key = _interpolate_track(tr["kf"])
            wins = [_face_window(r, crop_factor, align) for r in rows]
            wins = _smooth_windows(wins, window_size_mode, smooth_radius)
            track_list.append({
                "frames": frames, "rows": rows, "is_key": is_key, "wins": wins,
                "person": None, "emb": None, "sim": None, "ambiguous": False,
            })
        track_list.sort(key=lambda t: (t["frames"][0], t["rows"][0][0]))
        lines.append(f"{len(track_list)} track(s).")

        # ---- 3) identity of every track -------------------------------------
        for ti, tr in enumerate(track_list):
            cand = []
            for k, (f, r, key) in enumerate(zip(tr["frames"], tr["rows"], tr["is_key"])):
                if not key:
                    continue
                q = float(r[14]) * float(r[2] * r[3]) * (0.5 + 0.5 * tr["wins"][k]["frontality"])
                cand.append((q, f, r))
            cand.sort(key=lambda t: t[0], reverse=True)
            embs = []
            for _, f, r in cand[:max(1, int(embed_samples))]:
                embs.append(_safe_embed(_frame_bgr(images, f), r, sface_path))
            tr["emb"] = _mean_embedding(embs)

        # ---- 4) reference persons (external source images) ------------------
        person_ext = {}      # person -> embedding
        ref_faces = {}       # person -> crop tensor (1,S,S,3)
        for i in range(1, int(num_people) + 1):
            src = kwargs.get(f"source_image_{i}")
            if src is None:
                continue
            bgr = _frame_bgr(src, 0)
            rows = _detect_rows(bgr, yunet_path, score_threshold, 0, 1.0)
            if len(rows) == 0:
                msg = f"No face found in source_image_{i} - person {i} has no external reference."
                lines.append("WARNING: " + msg)
                _toast(node_id, "warn", "Mykee Face Tracks", msg)
                continue
            best = _pick_source_face(list(rows), source_which_face, source_face_index,
                                     (bgr.shape[1], bgr.shape[0]))
            if len(rows) > 1:
                lines.append(f"source_image_{i}: {len(rows)} faces found, using '{source_which_face}'"
                             + (f" #{int(source_face_index)}" if source_which_face in ("x_order", "y_order") else "") + ".")
            emb = _safe_embed(bgr, best, sface_path)
            if emb is None:
                lines.append(f"WARNING: could not embed the face of source_image_{i}.")
                continue
            person_ext[i] = emb
            crop_bgr, _ = _crop_face(bgr, best, size, crop_factor, align, edge_fill)
            ref_faces[i] = torch.from_numpy(_bgr_to_rgb_f32(crop_bgr)).unsqueeze(0)

        # ---- 5) assignment ---------------------------------------------------
        mode = assign_mode if assign_mode in _ASSIGN_MODES else "by_similarity"
        if mode != "by_similarity":
            # by position: the N best detections (quality = score^2 * sqrt(area), so tiny/low-score
            # false positives drop out), ordered along the chosen direction -> person 1, 2, ...
            def _quality(t):
                rr = [r for r, k in zip(t["rows"], t["is_key"]) if k] or list(t["rows"])
                return max(float(r[14]) ** 2 * math.sqrt(max(float(r[2] * r[3]), 1.0)) for r in rr)

            def _centre(t):
                return (float(np.mean([r[0] + r[2] / 2.0 for r in t["rows"]])),
                        float(np.mean([r[1] + r[3] / 2.0 for r in t["rows"]])))

            order_key = {
                "left_to_right": lambda c: c[0], "right_to_left": lambda c: -c[0],
                "top_to_bottom": lambda c: c[1], "bottom_to_top": lambda c: -c[1],
            }[mode]
            n_use = int(num_people)
            best_idx = sorted(range(len(track_list)), key=lambda i: _quality(track_list[i]), reverse=True)[:n_use]
            ordered = sorted(best_idx, key=lambda i: order_key(_centre(track_list[i])))
            for p, ti in enumerate(ordered, 1):
                tr = track_list[ti]
                tr["person"] = p
                if p in person_ext and tr["emb"] is not None:
                    tr["sim"] = float(_cosine_similarity(tr["emb"], person_ext[p]))
                cx, cy = _centre(tr)
                lines.append(f"Position order ({mode}): person {p} = face at x={cx:.0f} y={cy:.0f}.")
            if len(ordered) < n_use:
                lines.append(f"WARNING: only {len(ordered)} face(s) found for {n_use} person(s).")
        elif person_ext:
            assigned, sims, amb = _assign_tracks(
                [t["emb"] for t in track_list], person_ext, match_threshold, exclusive_matching
            )
            persons_sorted = sorted(person_ext)
            for ti, tr in enumerate(track_list):
                tr["person"] = assigned[ti]
                tr["ambiguous"] = amb[ti]
                if assigned[ti] is not None:
                    tr["sim"] = float(sims[ti, persons_sorted.index(assigned[ti])])
        elif int(num_people) == 1:
            for tr in track_list:  # nothing to compare with: everything is person 1
                tr["person"] = 1
            if track_list:
                lines.append("No source_image connected: every detected face is treated as person 1.")
        elif track_list:
            lines.append("WARNING: num_people > 1 but no source_image_N is connected - nothing can be assigned.")

        unmatched = [t for t in track_list if t["person"] is None]
        for t in track_list:
            if t["ambiguous"]:
                lines.append(
                    f"WARNING: track starting at frame {t['frames'][0]} is ambiguous between persons "
                    "(scores closer than %.2f) - check the result." % AMBIGUITY_MARGIN
                )

        # ---- 6) per-track geometry, items, self reference --------------------
        items = []
        person_tracks = {}
        track_meta = []
        for ti, tr in enumerate(track_list):
            first_item = len(items)
            if tr["person"] is not None:
                person_tracks.setdefault(tr["person"], []).append(ti)
                for k, f in enumerate(tr["frames"]):
                    w = tr["wins"][k]
                    mat = _window_matrix(w["cx"], w["cy"], w["side"], w["angle"], size)
                    items.append({
                        "person": tr["person"], "track": ti, "frame": int(f), "M": mat,
                        "geom": _face_geom_crop(tr["rows"][k], mat, w["lm_ok"]), "detected": bool(tr["is_key"][k]),
                        "row": tr["rows"][k].copy(), "skip": False,
                    })
            track_meta.append({
                "person": tr["person"], "frames": tr["frames"], "sim": tr["sim"],
                "item_range": (first_item, len(items)), "emb": tr["emb"],
            })

        # best in-image face per person
        self_faces = {}
        self_emb = {}
        self_item = {}
        for p, tis in person_tracks.items():
            best = None
            for ti in tis:
                tr = track_list[ti]
                for k, (f, r, key) in enumerate(zip(tr["frames"], tr["rows"], tr["is_key"])):
                    if not key:
                        continue
                    q = float(r[14]) * float(r[2] * r[3]) * (0.5 + 0.5 * tr["wins"][k]["frontality"])
                    if best is None or q > best[0]:
                        best = (q, ti, k, f, r)
            if best is None:
                continue
            _, ti, k, f, r = best
            bgr = _frame_bgr(images, f)
            crop_bgr, _ = _crop_face(bgr, r, size, crop_factor, align, edge_fill)
            self_faces[p] = torch.from_numpy(_bgr_to_rgb_f32(crop_bgr)).unsqueeze(0)
            self_emb[p] = _safe_embed(bgr, r, sface_path)
            start, _end = track_meta[ti]["item_range"]
            self_item[p] = start + k
            n_faces_p = sum(len(track_list[t]["frames"]) for t in tis)
            if skip_self_reference and n_frames == 1 and len(tis) >= 2 and n_faces_p >= 2:
                items[self_item[p]]["skip"] = True

        person_items = {}
        for idx, it in enumerate(items):
            if not it["skip"]:
                person_items.setdefault(it["person"], []).append(idx)

        # ---- 7) crops + masks -------------------------------------------------
        by_frame = {}
        slot = {}  # item index -> (person, position in that person's batch)
        crop_arrays = {}
        for p, idxs in person_items.items():
            if p > int(num_people):
                continue
            crop_arrays[p] = np.empty((len(idxs), size, size, 3), np.float32)
            for k, idx in enumerate(idxs):
                slot[idx] = (p, k)
                by_frame.setdefault(items[idx]["frame"], []).append(idx)
        for f in sorted(by_frame):
            bgr = _frame_bgr(images, f)
            for idx in by_frame[f]:
                p, k = slot[idx]
                crop_arrays[p][k] = _bgr_to_rgb_f32(_warp_to_crop(bgr, items[idx]["M"], size, edge_fill))
        outs = []
        gray = _gray_image(size)
        gray_mask = torch.zeros((1, size, size), dtype=torch.float32)
        for i in range(1, MAX_PEOPLE + 1):
            idxs = person_items.get(i, []) if i <= int(num_people) else []
            if idxs:
                crops_t = torch.from_numpy(crop_arrays[i])
                mask_t = torch.from_numpy(np.stack([
                    _ellipse_mask(size, items[j]["geom"], mask_grow_pct, mask_forehead_pct, mask_feather_pct)
                    for j in idxs
                ]))
            else:
                crops_t, mask_t = gray, gray_mask
            self_t = self_faces.get(i, gray)
            ref_t = ref_faces.get(i, self_t)
            outs.extend([crops_t, mask_t, ref_t, self_t])

        # ---- summary -----------------------------------------------------------
        for i in range(1, int(num_people) + 1):
            tis = person_tracks.get(i, [])
            n_crops = len(person_items.get(i, []))
            if not tis:
                lines.append(f"Person {i}: no face found" + ("" if i in person_ext else " (no external reference either)") + ".")
                continue
            sims_txt = ", ".join("%.2f" % track_list[t]["sim"] for t in tis if track_list[t]["sim"] is not None)
            lines.append(
                f"Person {i}: {len(tis)} track(s), {n_crops} crop(s)"
                + (f", similarity {sims_txt}" if sims_txt else "")
                + (", reference = source image" if i in person_ext else ", reference = best face in the input")
                + (", best face kept out of crops_%d as reference" % i if (i in self_item and items[self_item[i]]["skip"]) else "")
                + "."
            )
        for i in range(1, int(num_people) + 1):
            scales = [_scale_of(items[j]["M"]) for j in person_items.get(i, [])]
            if scales and min(scales) < 0.95:
                side_px = size / max(min(scales), 1e-6)
                lines.append(
                    f"WARNING: person {i}: the face window is {side_px:.0f}px in the input but crop_size is {size} "
                    f"(scale {min(scales):.2f}) - the crop is DOWNSCALED, so the refined face has less resolution "
                    f"than the original. For native detail set crop_size to about {int(math.ceil(side_px / 16.0) * 16)} "
                    "(and the H3 Reference Conditioning width/height with it), or lower crop_factor."
                )
        if unmatched and mode != "by_similarity":
            lines.append(
                f"{len(unmatched)} further track(s) are not among the {int(num_people)} best faces and are left untouched."
            )
        elif unmatched:
            lines.append(
                f"{len(unmatched)} track(s) matched nobody (below match_threshold {match_threshold:.2f}) and are left untouched."
            )

        tracks_obj = {
            "version": 1,
            "num_frames": n_frames,
            "frame_hw": (height, width),
            "crop_size": size,
            "edge_fill": edge_fill,
            "items": items,
            "person_items": person_items,
            "num_people": int(num_people),
            "tracks": track_meta,
            "person_emb": {p: (person_ext.get(p) if person_ext.get(p) is not None else self_emb.get(p))
                           for p in set(person_ext) | set(self_emb)},
            "person_emb_source": {p: ("external" if p in person_ext else "self")
                                  for p in set(person_ext) | set(self_emb)},
            "self_item": self_item,
            "ref_faces": {p: t for p, t in ref_faces.items()},
            "self_faces": {p: t for p, t in self_faces.items()},
        }
        summary = "\n".join(lines)
        return (summary, tracks_obj, *outs)


# ---------------------------------------------------------------------------
# Node 2: Mykee H3 Face Latent
# ---------------------------------------------------------------------------

class MykeeH3FaceLatent:
    """
    MiniMax H3 only. Turns face crops into H3 video+audio latents, one latent
    per crop (the output is a ComfyUI list, so your sampler runs once per
    crop with batch size 1 - which is what H3's conditioning expects).

    How: H3's sampler works on a nested latent (24-channel video at 1/16
    resolution + an audio latent). A still is a 5-frame clip, so every crop is
    repeated 5 times, encoded with the H3 video VAE (the same recipe H3's own
    Fun ControlNet uses), and put next to the audio part of the template
    latent. No audio VAE is needed.

    Wire `av_latent` from an *H3 Reference Conditioning* node set up for the
    refine pass: width = height = crop_size, length = 5, prompt = a short
    face prompt, ref images = ref_face_N / self_face_N of the Face Tracks
    node. Use the same positive conditioning in the sampler, and a denoise of
    roughly 0.35-0.5 in BasicScheduler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "crops": ("IMAGE", {"tooltip": "crops_N from Mykee Face Tracks."}),
            "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
            "av_latent": ("LATENT", {"tooltip": "LATENT output of an H3 Reference Conditioning node with width = height = crop_size and length = 5. Only its shape and audio part are used."}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, crops, vae, av_latent):
        samples = av_latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError(
                "av_latent is not a MiniMax H3 video+audio latent. Connect the LATENT output of an "
                "H3 Reference Conditioning node (length 5)."
            )
        video_t, audio_t = samples.tensors[0], samples.tensors[1]
        _, _, lat_t, lat_h, lat_w = video_t.shape
        frame_count = max((lat_t - 2) // 5, 0) * 17 + 5  # inverse of H3's frame -> latent mapping
        target_h, target_w = lat_h * 16, lat_w * 16
        if crops.shape[1] != target_h or crops.shape[2] != target_w:
            raise ValueError(
                f"Crop size {crops.shape[2]}x{crops.shape[1]} does not match the H3 latent "
                f"{target_w}x{target_h}. Set the H3 Reference Conditioning width/height to the "
                "Face Tracks crop_size."
            )
        import comfy.nested_tensor as nested_tensor

        extra = {k: v for k, v in av_latent.items() if k not in ("samples", "noise_mask")}
        out = []
        for k in range(crops.shape[0]):
            frames = crops[k:k + 1, :, :, :3].expand(frame_count, -1, -1, -1).contiguous()
            z = vae.encode(frames)
            if tuple(z.shape) != tuple(video_t.shape):
                raise ValueError(
                    f"The H3 VAE gave a latent of shape {tuple(z.shape)}, expected {tuple(video_t.shape)}. "
                    "Check that the connected VAE is the MiniMax H3 video VAE and av_latent has length 5."
                )
            lat = dict(extra)
            lat["samples"] = nested_tensor.NestedTensor((z.float(), audio_t.clone()))
            out.append(lat)
        return (out,)


# ---------------------------------------------------------------------------
# Node 3: Mykee Face Paste
# ---------------------------------------------------------------------------

def _match_color(refined, orig, mask, strength):
    """Moves the refined crop's per-channel LAB mean/std (measured inside the
    face mask) toward the original's, so the pasted face has the same tone as
    its surroundings."""
    core = mask > 0.5
    if int(core.sum()) < 200 or strength <= 0:
        return refined
    r_lab = cv2.cvtColor(refined.astype(np.float32), cv2.COLOR_RGB2LAB)
    o_lab = cv2.cvtColor(orig.astype(np.float32), cv2.COLOR_RGB2LAB)
    out = r_lab.copy()
    for ch in range(3):
        rv, ov = r_lab[..., ch][core], o_lab[..., ch][core]
        mu_r, mu_o = float(rv.mean()), float(ov.mean())
        sd_r, sd_o = float(rv.std()), float(ov.std())
        gain = float(np.clip(sd_o / sd_r, 0.8, 1.25)) if sd_r > 1e-3 else 1.0
        out[..., ch] = (r_lab[..., ch] - mu_r) * gain + mu_o
    out = r_lab + (out - r_lab) * float(strength)
    res = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(res, 0.0, 1.0)


def _paste_crop(frame, refined_rgb, mask, matrix, size):
    """Blends one refined crop into `frame` (float32 HxWx3 RGB, in place) by
    mapping it back through the inverse of the stored transform."""
    h, w = frame.shape[:2]
    scale = _scale_of(matrix)
    sp = max(2, int(round(size / scale)))
    f = sp / float(size)
    interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC
    ref_s = cv2.resize(refined_rgb, (sp, sp), interpolation=interp)
    m_s = cv2.resize(mask, (sp, sp), interpolation=interp)
    ms = matrix.copy()
    ms *= f  # original -> scaled-crop coordinates
    x0, y0, x1, y1 = _window_roi(ms, sp, w, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    m_roi = ms.copy()
    m_roi[:, 2] += ms[:, :2] @ np.array([x0, y0], dtype=np.float64)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    wr = cv2.warpAffine(ref_s, m_roi, (x1 - x0, y1 - y0), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    wm = cv2.warpAffine(m_s, m_roi, (x1 - x0, y1 - y0), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = np.clip(wm, 0.0, 1.0)[..., None]
    roi = frame[y0:y1, x0:x1]
    roi *= (1.0 - wm)
    roi += wr * wm


def _composite_item(frame_bgr, it, refined_rgb, size, grow, fore, feather, color_match, color_strength, edge_fill):
    """Pastes ONE refined crop into a copy of `frame_bgr` and returns the
    composite as uint8 BGR. Used by the identity guard so it can measure the
    face exactly as it will look in the final image."""
    frame = _bgr_to_rgb_f32(frame_bgr)
    mask = _ellipse_mask(size, it["geom"], grow, fore, feather)
    ref = refined_rgb
    if color_match != "off" and color_strength > 0:
        o_crop = _bgr_to_rgb_f32(_warp_to_crop(frame_bgr, it["M"], size, edge_fill))
        ref = _match_color(ref, o_crop, mask, color_strength)
    _paste_crop(frame, ref, mask, it["M"], size)
    return _rgb_f32_to_bgr_u8(frame)


def _detail(bgr, row):
    """Sharpness (variance of the Laplacian) of the face box, for the
    resolution-loss check."""
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(row[0])), max(0, int(row[1]))
    x1, y1 = min(w, int(row[0] + row[2])), min(h, int(row[1] + row[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _label(bgr, text):
    out = bgr.copy()
    scale = max(0.5, out.shape[0] / 900.0)
    cv2.rectangle(out, (0, 0), (out.shape[1], int(34 * scale) + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (6, int(26 * scale) + 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class MykeeFacePaste:
    """
    Blends refined face crops back into the original image / video frames.

    `refined_N` must hold the refined version of `crops_N` - same count and
    order. It can be one image batch (normal models) or a list (e.g. the
    per-crop results of the H3 chain; if every list entry has several frames,
    `frame_pick` chooses which one is the face).

    Identity guard: before anything is pasted, every track's original and
    refined face are compared (SFace) with the person's reference. The refined
    track is only used if it is not less similar than the original, minus
    `guard_margin`. Decided per TRACK (never per frame), so a video does not
    flicker between refined and original. Identity is judged against
    source_image_N when you connected one, otherwise against the best face in
    the input.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        yunet_files = _list_annotator_files(".onnx") or ["face_detection_yunet_2023mar.onnx"]
        sface_files = _list_annotator_files(".onnx") or ["face_recognition_sface_2021dec.onnx"]
        required = {
            "images": ("IMAGE", {"tooltip": "The original image / video frames (the same input as Mykee Face Tracks)."}),
            "tracks": ("FACE_TRACKS",),
            "num_people": ("INT", {"default": 1, "min": 1, "max": MAX_PEOPLE, "step": 1, "tooltip": "Not used any more (the number of people comes from Mykee Face Tracks). Kept so that saved workflows keep loading; hidden in the UI."}),
            "model_file_yunet": (yunet_files, {"default": _default_annotator_choice(yunet_files, "face_detection_yunet_2023mar.onnx")}),
            "model_file_sface": (sface_files, {"default": _default_annotator_choice(sface_files, "face_recognition_sface_2021dec.onnx")}),
            "mask_grow_pct": ("FLOAT", {"default": 10.0, "min": -50.0, "max": 200.0, "step": 1.0, "tooltip": "Size of the blend mask relative to the detected face box."}),
            "mask_forehead_pct": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            "mask_feather_pct": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Softness of the blend edge, as a percentage of the crop size."}),
            "color_match": (["lab_mean_std", "off"], {
                "default": "lab_mean_std",
                "tooltip": "Matches the refined face's tone (LAB mean/std inside the mask) to the original so no colour seam shows.",
            }),
            "color_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "identity_guard": ("BOOLEAN", {
                "default": True,
                "tooltip": "Only paste a refined track if its face is not less similar to the reference than the original was.",
            }),
            "guard_margin": ("FLOAT", {"default": 0.02, "min": -0.5, "max": 0.5, "step": 0.01, "tooltip": "Allowed drop in SFace similarity before a refined track is rejected."}),
            "guard_samples": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1, "tooltip": "How many frames per track are compared (video)."}),
            "score_threshold": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.99, "step": 0.01, "tooltip": "Detector threshold used to find the face in a refined crop (guard)."}),
            "frame_pick": (FRAME_PICKS, {"default": "middle", "tooltip": "If a refined entry has several frames (H3 returns 5), which one is the face."}),
            "guard_measure": (["in_place", "crop"], {
                "default": "in_place",
                "tooltip": "in_place: the guard compares the face in the ORIGINAL frame with the face in the finished composite, at the original resolution (what you will actually see, same as Identity Compare). crop: compares the resampled crops (older behaviour).",
            }),
            "min_detail_ratio": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                "tooltip": "EXPERIMENTAL, off by default. in_place guard only: reject a track when the pasted face has less than this fraction of the original face's pixel-level Laplacian sharpness. That measure is dominated by pixel-level grain, so it can reject refinements that look sharper to the eye; leave it at 0 unless you have seen a clearly blurry refinement slip through.",
            }),
        }
        optional = {
            "refined_all": ("IMAGE", {
                "tooltip": "The refined results for ALL crops of Mykee Face Tracks, in the same order (person 1, 2, ...): the decoded output of your refine chain.",
            }),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "summary", "before_after")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @staticmethod
    def _collect(refined, count, frame_pick):
        """Normalises a refined_N input (batch or list of batches) to a list of
        exactly `count` HxWx3 float32 RGB arrays."""
        tensors = [t for t in refined if t is not None]
        if not tensors:
            return None
        total = sum(int(t.shape[0]) for t in tensors)
        out = []
        if total == count:
            for t in tensors:
                for k in range(t.shape[0]):
                    out.append(t[k, :, :, :3].detach().cpu().float().numpy())
            return out
        if len(tensors) == count:
            for t in tensors:
                n = int(t.shape[0])
                k = {"first": 0, "last": n - 1}.get(frame_pick, n // 2)
                out.append(t[k, :, :, :3].detach().cpu().float().numpy())
            return out
        raise ValueError(
            f"refined input has {total} image(s) in {len(tensors)} entr(ies), but {count} crop(s) were expected."
        )

    def run(self, images, tracks, num_people, model_file_yunet, model_file_sface, mask_grow_pct,
            mask_forehead_pct, mask_feather_pct, color_match, color_strength, identity_guard,
            guard_margin, guard_samples, score_threshold, frame_pick,
            guard_measure="in_place", min_detail_ratio=0.0, **kwargs):
        _require_cv2()
        images_t = images[0] if len(images) == 1 else torch.cat(list(images), dim=0)
        tr = _first(tracks)
        # the number of people comes from the tracks; the widget is kept only so
        # that saved workflows keep loading
        num_people = int(tr.get("num_people", _first(num_people)))
        grow, fore, feather = float(_first(mask_grow_pct)), float(_first(mask_forehead_pct)), float(_first(mask_feather_pct))
        color_match, color_strength = _first(color_match), float(_first(color_strength))
        guard_on = bool(_first(identity_guard))
        margin, n_samples = float(_first(guard_margin)), int(_first(guard_samples))
        det_thr = float(_first(score_threshold))
        in_place = _first(guard_measure) == "in_place"
        min_detail = float(_first(min_detail_ratio))
        frame_pick = _first(frame_pick)
        yunet_path = _resolve_model_path(_first(model_file_yunet))
        sface_path = _resolve_model_path(_first(model_file_sface))

        size = int(tr["crop_size"])
        items = tr["items"]
        n_frames = images_t.shape[0]
        lines = []
        if n_frames != tr["num_frames"] or tuple(images_t.shape[1:3]) != tuple(tr["frame_hw"]):
            raise ValueError(
                f"images ({n_frames} x {images_t.shape[2]}x{images_t.shape[1]}) do not match the frames that "
                f"Mykee Face Tracks analysed ({tr['num_frames']} x {tr['frame_hw'][1]}x{tr['frame_hw'][0]})."
            )

        # ---- gather the refined crops, per item --------------------------------
        refined = {}
        all_slices = {}
        all_list = kwargs.get("refined_all")
        if all_list is not None:
            # refined_all covers every person of the tracks, whatever the num_people widget says
            num_people = max(num_people, int(tr.get("num_people", num_people)))
            order = [(p, tr["person_items"].get(p, [])) for p in range(1, int(tr.get("num_people", MAX_PEOPLE)) + 1)]
            order = [(p, idxs) for p, idxs in order if idxs]
            total = sum(len(idxs) for _, idxs in order)
            if total:
                arrs_all = self._collect(all_list, total, frame_pick)
                off = 0
                for p, idxs in order:
                    all_slices[p] = arrs_all[off:off + len(idxs)]
                    off += len(idxs)
        for p in range(1, num_people + 1):
            idxs = tr["person_items"].get(p, [])
            lst = kwargs.get(f"refined_{p}")
            if idxs and lst is None and p in all_slices:
                arrs = all_slices[p]
                for idx, arr in zip(idxs, arrs):
                    if arr.shape[0] != size or arr.shape[1] != size:
                        interp = cv2.INTER_AREA if arr.shape[0] > size else cv2.INTER_CUBIC
                        arr = cv2.resize(arr, (size, size), interpolation=interp)
                    refined[idx] = np.ascontiguousarray(arr, dtype=np.float32)
                continue
            if not idxs or lst is None:
                if idxs and lst is None:
                    lines.append(f"Person {p}: {len(idxs)} crop(s) but refined_{p} is not connected - left untouched.")
                continue
            arrs = self._collect(lst, len(idxs), frame_pick)
            for idx, arr in zip(idxs, arrs):
                if arr.shape[0] != size or arr.shape[1] != size:
                    interp = cv2.INTER_AREA if arr.shape[0] > size else cv2.INTER_CUBIC
                    arr = cv2.resize(arr, (size, size), interpolation=interp)
                refined[idx] = np.ascontiguousarray(arr, dtype=np.float32)

        # ---- per-track decisions (identity guard) --------------------------------
        track_items = {}
        for idx in refined:
            track_items.setdefault(items[idx]["track"], []).append(idx)
        accepted_tracks = set()
        montage = []
        out_dtype = images_t.dtype
        edge_fill = tr.get("edge_fill", "replicate")
        frame_cache = {}

        def get_bgr(f):
            if f not in frame_cache:
                frame_cache.clear() if len(frame_cache) > 4 else None
                frame_cache[f] = _frame_bgr(images_t, f)
            return frame_cache[f]

        for ti in sorted(track_items):
            idxs = sorted(track_items[ti], key=lambda j: items[j]["frame"])
            person = items[idxs[0]]["person"]
            emb = tr["person_emb"].get(person)
            decision, note = True, ""
            sample = idxs
            if in_place:
                # measure only on frames where the detector really found the face
                # (interpolated frames have no real face to compare in the original)
                real = [j for j in idxs if items[j].get("detected", True)]
                if real:
                    sample = real
            if len(sample) > n_samples:
                pick = np.linspace(0, len(sample) - 1, n_samples).round().astype(int)
                sample = [sample[k] for k in pick]
            before, after, undetected = [], [], 0
            detail_ratios = []
            first_pair = None
            for j in sample:
                it = items[j]
                orig_bgr = _warp_to_crop(get_bgr(it["frame"]), it["M"], size, edge_fill)
                ref_bgr = _rgb_f32_to_bgr_u8(refined[j])
                if first_pair is None:
                    first_pair = (orig_bgr, ref_bgr, j)
                if not (guard_on and emb is not None):
                    continue
                is_self = (tr["person_emb_source"].get(person) == "self"
                           and tr["self_item"].get(person) == j)
                if is_self:
                    continue
                if in_place:
                    full = get_bgr(it["frame"])
                    e_before = _safe_embed(full, it["row"], sface_path)
                    comp = _composite_item(full, it, refined[j].copy(), size, grow, fore, feather,
                                           color_match, color_strength, edge_fill)
                    rows = _detect_rows(comp, yunet_path, det_thr, 0, 1.0)
                    e_after = None
                    if len(rows):
                        oc = np.array([it["row"][0] + it["row"][2] / 2.0, it["row"][1] + it["row"][3] / 2.0])
                        best = min(rows, key=lambda r: float(np.hypot(*(np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - oc))))
                        if float(np.hypot(*(np.array([best[0] + best[2] / 2.0, best[1] + best[3] / 2.0]) - oc))) <= 0.5 * max(it["row"][2], it["row"][3]):
                            e_after = _safe_embed(comp, best, sface_path)
                    d0, d1 = _detail(full, it["row"]), _detail(comp, it["row"])
                    if d0 and d1 is not None and d0 > 1e-6:
                        detail_ratios.append(d1 / d0)
                else:
                    e_before = _safe_embed(orig_bgr, _row_in_crop(it["row"], it["M"]), sface_path)
                    rows = _detect_rows(ref_bgr, yunet_path, det_thr, 0, 1.0)
                    e_after = None
                    if len(rows):
                        centre = np.array([size / 2.0, size / 2.0])
                        best = min(rows, key=lambda r: float(np.hypot(*(np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - centre))))
                        e_after = _safe_embed(ref_bgr, best, sface_path)
                if e_before is not None:
                    before.append(_cosine_similarity(e_before, emb))
                if e_after is None:
                    undetected += 1
                else:
                    after.append(_cosine_similarity(e_after, emb))

            if guard_on and emb is not None and (before or after or undetected):
                if undetected and not after:
                    decision, note = False, "no face could be found in the refined crop(s)"
                elif after and before:
                    mb, ma = float(np.mean(before)), float(np.mean(after))
                    decision = ma >= mb - margin
                    note = f"similarity {mb:.3f} -> {ma:.3f}"
                elif after:
                    note = f"similarity after {float(np.mean(after)):.3f} (original not measurable)"
            elif guard_on and emb is None:
                note = "no reference embedding - guard skipped"
            if guard_on and in_place and decision and min_detail > 0 and detail_ratios:
                dr = float(np.mean(detail_ratios))
                note += ("; " if note else "") + f"detail {dr:.2f}x"
                if dr < min_detail:
                    decision = False
                    note += f" < {min_detail:.2f} (refined face is softer than the original)"
            if undetected and after:
                note += f" ({undetected} refined crop(s) without a detectable face)"

            if decision:
                accepted_tracks.add(ti)
            lines.append(
                f"Track {ti} (person {person}, {len(idxs)} frame(s)): "
                + ("PASTED" if decision else "REJECTED (kept original)")
                + (f" - {note}" if note else "")
            )
            if first_pair is not None:
                o, r, j = first_pair
                tag = "PASTED" if decision else "REJECTED"
                pair = np.concatenate([_label(o, "original"), _label(r, f"refined - {tag}")], axis=1)
                montage.append(_bgr_to_rgb_f32(pair))

        # ---- compositing -----------------------------------------------------------
        out = images_t.clone().float()
        by_frame = {}
        for idx in refined:
            if items[idx]["track"] in accepted_tracks:
                by_frame.setdefault(items[idx]["frame"], []).append(idx)
        pasted = 0
        for f, idxs in by_frame.items():
            frame = np.ascontiguousarray(out[f][..., :3].numpy())
            for j in idxs:
                it = items[j]
                mask = _ellipse_mask(size, it["geom"], grow, fore, feather)
                ref = refined[j]
                if color_match != "off" and color_strength > 0:
                    # the original (unmodified) frame is the colour reference
                    o_crop = _bgr_to_rgb_f32(_warp_to_crop(get_bgr(f), it["M"], size, edge_fill))
                    ref = _match_color(ref, o_crop, mask, color_strength)
                _paste_crop(frame, ref, mask, it["M"], size)
                pasted += 1
            out[f][..., :3] = torch.from_numpy(frame)
        lines.insert(0, f"{pasted} face crop(s) pasted into {len(by_frame)} frame(s).")

        if montage:
            m_t = torch.from_numpy(np.stack(montage))
        else:
            m_t = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
        return (out.to(out_dtype), "\n".join(lines), m_t)


class MykeeImageBatchToList:
    """Turns an image batch into a LIST, so everything connected after it
    (conditioning with reference images, VAE Encode, sampler, VAE Decode) runs
    once per entry instead of once for the whole batch. Needed for models whose
    conditioning takes the crop as an image (Qwen Image Edit, ...): a batch of
    crops would otherwise be treated as several reference images of ONE
    generation and come back as a single image. Mykee Face Paste accepts the
    resulting list of results directly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Usually crops_N of Mykee Face Tracks."}),
                "chunk_size": ("INT", {
                    "default": 1, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Images per list entry. 1 = every image is processed on its own (needed when the crop goes into the conditioning). More = faster for plain VAE Encode -> sampler chains without image conditioning.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, chunk_size):
        n = int(images.shape[0])
        step = max(1, int(chunk_size))
        return ([images[i:i + step] for i in range(0, n, step)],)


NODE_CLASS_MAPPINGS = {
    "MykeeFaceTracks": MykeeFaceTracks,
    "MykeeH3FaceLatent": MykeeH3FaceLatent,
    "MykeeFacePaste": MykeeFacePaste,
    "MykeeImageBatchToList": MykeeImageBatchToList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeFaceTracks": "Mykee Face Tracks (detect + crop)",
    "MykeeH3FaceLatent": "Mykee H3 Face Latent",
    "MykeeFacePaste": "Mykee Face Paste (blend back)",
    "MykeeImageBatchToList": "Mykee Image Batch To List",
}
