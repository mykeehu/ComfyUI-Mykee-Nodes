"""Face mesh helpers shared by the Face Refine nodes (edge align).

MediaPipe Face Landmarker (478 points, optional dependency) with its model
download, the mesh group indices, and the smoothing / thin-plate-spline
helpers used to move a face outline. No node is registered here.
"""

import os

import numpy as np

from .mykee_character import cv2, _annotators_dir

LANDMARKER_FILE = "face_landmarker.task"

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

LANDMARKER_MAX_BYTES = 64 * 1024 * 1024

MESH_FALLBACK = {
    "FACE_LANDMARKS_FACE_OVAL": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                                 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                                 54, 103, 67, 109],
    "FACE_LANDMARKS_LEFT_EYE": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "FACE_LANDMARKS_RIGHT_EYE": [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "FACE_LANDMARKS_LEFT_EYEBROW": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "FACE_LANDMARKS_RIGHT_EYEBROW": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "FACE_LANDMARKS_NOSE": [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294,
                            326, 327, 344, 440],
    "FACE_LANDMARKS_LIPS": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269,
                            270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13,
                            312, 311, 310, 415],
}

def _blur(mask, sigma):
    if sigma <= 0.5:
        return mask
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), sigma)

def _conn_idx(conns):
    out = set()
    for c in conns:
        a, b = (c.start, c.end) if hasattr(c, "start") else (c[0], c[1])
        out.add(int(a))
        out.add(int(b))
    return sorted(out)

class _FaceMesh:
    """478-point face mesh (x, y in pixels, z on the x scale). Every mesh has
    the same points in the same order, so two faces can be compared and
    regions cut point by point. Optional: needs the mediapipe package."""

    _cache = {}

    def __init__(self, choice):
        self.ok = False
        self.error = None
        self.groups = {}
        if choice != "mediapipe":
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except Exception as e:
            self.error = ("mediapipe is not installed (pip install mediapipe --no-deps absl-py flatbuffers "
                          f"- --no-deps keeps it from adding a second OpenCV): {type(e).__name__}")
            return
        try:
            path = self._model_path()
            if path not in self._cache:
                opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=path), num_faces=6,
                                                    min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
                self._cache.clear()
                self._cache[path] = vision.FaceLandmarker.create_from_options(opts)
            self.landmarker = self._cache[path]
            self.mp = mp
            C = getattr(vision, "FaceLandmarksConnections", None)

            def group(*names):
                # the package's own connection lists where this version has
                # them, otherwise the fixed MediaPipe face-mesh indices
                out = []
                for nm in names:
                    conns = getattr(C, nm, None) if C is not None else None
                    out += _conn_idx(conns) if conns else MESH_FALLBACK[nm]
                return out

            self.groups = {
                "oval": group("FACE_LANDMARKS_FACE_OVAL"),
                "eyes": group("FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE"),
                "brows": group("FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW"),
                "nose": group("FACE_LANDMARKS_NOSE"),
                "lips": group("FACE_LANDMARKS_LIPS"),
            }
            self.ok = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _model_path():
        base = os.path.realpath(_annotators_dir())
        path = os.path.join(base, LANDMARKER_FILE)
        if os.path.isfile(path):
            return path
        import urllib.request
        tmp = path + ".part"
        with urllib.request.urlopen(LANDMARKER_URL, timeout=60) as r:
            data = r.read(LANDMARKER_MAX_BYTES + 1)
        # a real model bundle is a few MB of binary data; an error / login /
        # proxy page is small text or HTML
        head = data[:64].lstrip().lower()
        if (len(data) > LANDMARKER_MAX_BYTES or len(data) < 512 * 1024
                or head.startswith((b"<", b"{")) or b"html" in head):
            raise RuntimeError(
                f"the download of {LANDMARKER_FILE} did not return the model ({len(data)} bytes, starts with "
                f"{data[:16]!r}). Download it by hand from {LANDMARKER_URL} and put it into models/annotators.")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path

    def landmarks(self, arr, near_row=None):
        """(478, 3) points of the face closest to near_row (YuNet row), or the
        largest face; None if no face."""
        if not self.ok:
            return None
        h, w = arr.shape[:2]
        try:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)))
            res = self.landmarker.detect(img)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return None
        faces = []
        for lms in res.face_landmarks or []:
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            faces.append(pts)
        if not faces:
            return None
        if near_row is not None:
            cx, cy = float(near_row[0] + near_row[2] / 2.0), float(near_row[1] + near_row[3] / 2.0)
            return min(faces, key=lambda q: float(np.hypot(q[:, 0].mean() - cx, q[:, 1].mean() - cy)))
        return max(faces, key=lambda q: float(np.ptp(q[:, 0]) * np.ptp(q[:, 1])))

    def hull_mask(self, h, w, pts, groups, grow=0.0, feather=0.0):
        """Filled convex hull of the given landmark groups, optionally grown
        and feathered (both as a fraction of the face width)."""
        idx = [i for g in groups for i in self.groups.get(g, []) if i < len(pts)]
        if len(idx) < 3:
            return None
        hull = cv2.convexHull(np.round(pts[idx, :2]).astype(np.int32))
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        fw = float(np.ptp(pts[:, 0])) or 1.0
        if grow > 0:
            k = max(3, int(fw * grow)) | 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        m = m.astype(np.float32)
        return _blur(m, fw * feather) if feather > 0 else m

def _tps_field(ctrl, vals, H, W, step=4, reg=1e-4):
    """Thin-plate spline through the control points ctrl (N, 2) with the 2D
    values vals (N, 2), evaluated over an H x W picture: the smoothest field
    (least bending) that passes through every point. Evaluated on a coarser
    grid (every step pixels) and scaled up - the field is smooth anyway."""
    c = ctrl.astype(np.float64)
    mu = c.mean(0)
    sc = float(np.ptp(c, axis=0).max()) or 1.0
    cn = (c - mu) / sc                                  # well-conditioned units

    def U(r2):
        return np.where(r2 > 1e-12, 0.5 * r2 * np.log(np.maximum(r2, 1e-12)), 0.0)

    n = len(cn)
    d2 = ((cn[:, None, :] - cn[None, :, :]) ** 2).sum(2)
    K = U(d2) + reg * np.eye(n)
    P = np.concatenate([np.ones((n, 1)), cn], 1)
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = vals
    coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
    w_, a_ = coef[:n], coef[n:]
    gh, gw = (H + step - 1) // step + 1, (W + step - 1) // step + 1
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64) * step
    g = (np.stack([gx.ravel(), gy.ravel()], 1) - mu) / sc
    out = np.zeros((len(g), 2))
    for i in range(0, len(g), 8192):                    # bounded memory
        gg = g[i:i + 8192]
        r2 = ((gg[:, None, :] - cn[None, :, :]) ** 2).sum(2)
        out[i:i + 8192] = U(r2) @ w_ + a_[0] + gg @ a_[1:]
    fx = cv2.resize(out[:, 0].reshape(gh, gw).astype(np.float32), ((gw - 1) * step + 1, (gh - 1) * step + 1),
                    interpolation=cv2.INTER_LINEAR)[:H, :W]
    fy = cv2.resize(out[:, 1].reshape(gh, gw).astype(np.float32), ((gw - 1) * step + 1, (gh - 1) * step + 1),
                    interpolation=cv2.INTER_LINEAR)[:H, :W]
    return fx, fy

def _soft_outside(mask, sigma):
    """1 on the mask and a little beyond it, then falling smoothly to 0 -
    with no step at the mask's edge: the mask is grown by 2.5 sigma before
    the Gaussian, so the blurred value is already ~1 at the original edge."""
    r = max(1, int(round(2.5 * sigma)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate((mask > 0.5).astype(np.uint8), k).astype(np.float32)
    return np.maximum(mask.astype(np.float32), _blur(grown, max(0.5, sigma)))

def _smooth_over_mesh(pts, vals, radius, k=8):
    """Averages per-point values with their k nearest mesh neighbours,
    repeated until the averaging reaches about radius (a diffusion: the
    reach grows with the square root of the rounds)."""
    p = pts[:, :2].astype(np.float64)
    n = len(p)
    if n <= k:
        return vals
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(2)
    nb = np.argsort(d2, axis=1)[:, 1:k + 1]
    spacing = float(np.sqrt(np.median(d2[np.arange(n)[:, None], nb])))
    rounds = int(np.clip(np.ceil((radius / max(1e-6, spacing)) ** 2), 1, 200))
    v = vals.astype(np.float64).copy()
    for _ in range(rounds):
        v = 0.5 * v + 0.5 * v[nb].mean(1)
    return v.astype(np.float32)
