"""
ComfyUI-Mykee-Nodes / character-consistency nodes

A ComfyUI-native rebuild of Inline Studio's "Encode Character" logic:
  1) MykeeFaceDetectAlignCrop     - YuNet face detection, an aligned
                                     (SFace-compatible) crop, and a
                                     context (padded) crop, plus a face mask
  2) MykeeFaceEmbedSFace          - SFace 128-dim face embedding from the
                                     aligned crop
  3) MykeeSubjectEmbedDINOv2      - DINOv2 768-dim "subject" (body/clothing)
                                     embedding from the full reference image
  4) MykeeIdentityScore           - cosine similarity between two embeddings
                                     (face or subject embeddings), with a
                                     pass/fail threshold - this mirrors
                                     Inline Studio's "score", used to
                                     filter/rank generated takes
  5) MykeeCharacterReferencePack  - merges 1-9 reference images into one
                                     IMAGE batch and builds the native
                                     MiniMax H3 <Picture N> prompt prefix
  6) MykeeIdentityCompare         - all-in-one convenience node: compares a
                                     generated image against up to 10 source
                                     images directly (live 1-10 person-count
                                     switch), one similarity/annotated-image
                                     output per source image

The models (YuNet, SFace, DINOv2-base) are the same ones Inline Studio
uses, and all are freely redistributable (MIT / Apache-2.0):
  - face_detection_yunet_2023mar.onnx   (OpenCV Zoo)
  - face_recognition_sface_2021dec.onnx (OpenCV Zoo)
  - facebook/dinov2-base                (HuggingFace transformers)
"""

import os
import numpy as np
import torch

try:
    import cv2
    # Some opencv-python builds (this crash has been seen on recent 4.x
    # Windows wheels) have a thread-safety bug in cv2.dnn's shape-inference
    # cache (Net::Impl::getLayerShapesRecursively) that intermittently
    # raises a NaryEltwiseLayer 'findCommonShape' assertion when a
    # dnn-backed model (FaceRecognizerSF/YuNet) has its .feature()/.detect()
    # called more than once per process - which MykeeIdentityCompare now
    # does routinely (once per detected face, possibly 10+ times per run).
    # Restricting OpenCV to a single thread avoids the internal race.
    cv2.setNumThreads(1)
except ImportError:
    cv2 = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


def _toast(node_id, severity, summary, detail):
    """Sends a non-blocking ComfyUI toast notification (the same
    top-right-corner UI ComfyUI itself uses for its own messages) instead
    of the node having to raise a hard error that aborts the WHOLE prompt
    over one missing face. The JS-side listener (web/mykee_toast.js) turns
    this into an app.extensionManager.toast.add(...) call. Safe no-op if
    there's no running PromptServer (e.g. outside a real ComfyUI process)."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync(
            "mykee.toast",
            {"node": node_id, "severity": severity, "summary": summary, "detail": detail, "life": 6000},
        )
    except Exception:
        pass  # a failed notification must never break the actual node


# ---------------------------------------------------------------------------
# Helper types / helper functions
# ---------------------------------------------------------------------------

class AnyType(str):
    """Wildcard type - compatible with any embedding-output socket."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")

ANNOTATOR_SUBDIR = "annotators"


def _annotators_dir():
    """Where models are looked up: ComfyUI/models/annotators if available,
    otherwise a ./models/annotators folder next to this package."""
    if folder_paths is not None:
        try:
            base = os.path.join(folder_paths.models_dir, ANNOTATOR_SUBDIR)
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pack root (this file lives in nodes/)
    base = os.path.join(here, "models", ANNOTATOR_SUBDIR)
    os.makedirs(base, exist_ok=True)
    return base


def _resolve_model_path(name_or_path):
    """If given a plain filename, looks it up under the annotators folder;
    if given a full/relative path that exists, uses that instead."""
    if os.path.isfile(name_or_path):
        return name_or_path
    candidate = os.path.join(_annotators_dir(), name_or_path)
    if os.path.isfile(candidate):
        return candidate
    return name_or_path  # let the caller raise a sensible error


def _list_annotator_files(*exts):
    base = _annotators_dir()
    out = []
    try:
        for f in sorted(os.listdir(base)):
            if f.lower().endswith(exts):
                out.append(f)
    except FileNotFoundError:
        pass
    return out


def _list_annotator_dirs():
    base = _annotators_dir()
    out = []
    try:
        for f in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, f)):
                out.append(f)
    except FileNotFoundError:
        pass
    return out


def _default_annotator_choice(candidates, preferred_name):
    """Picks preferred_name out of a combo widget's candidate list, so the
    widget defaults to the CORRECT, expected model regardless of what else
    is sitting in the annotators folder or how the filenames happen to
    sort alphabetically (a combo widget with no explicit 'default'
    defaults to whichever file sorts first - which can silently be the
    wrong model, and is exactly how the YuNet and SFace fields on
    Identity Compare once ended up both pointing at the same file by
    accident). Falls back to the first candidate if preferred_name isn't
    present yet (e.g. model not downloaded)."""
    if preferred_name in candidates:
        return preferred_name
    return candidates[0] if candidates else preferred_name


def _edge_fill_cv2_params(edge_fill):
    """Maps the shared 'edge_fill' choice (replicate/reflect/constant_gray/
    constant_black) to cv2.copyMakeBorder's (borderType, value) args - used
    both by context_crop's padding and ffhq_crop's padding below, so the
    two stay consistent."""
    return {
        "replicate": (cv2.BORDER_REPLICATE, None),
        "reflect": (cv2.BORDER_REFLECT_101, None),
        "constant_gray": (cv2.BORDER_CONSTANT, (127, 127, 127)),
        "constant_black": (cv2.BORDER_CONSTANT, (0, 0, 0)),
    }[edge_fill]


def tensor_to_bgr(img_tensor):
    """ComfyUI IMAGE (1,H,W,C float 0..1 RGB) -> uint8 BGR numpy (H,W,C)."""
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_tensor(bgr):
    """uint8 BGR numpy (H,W,C) -> ComfyUI IMAGE (1,H,W,C float 0..1 RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def mask_to_tensor(mask_u8):
    """uint8 (H,W) 0..255 -> ComfyUI MASK (1,H,W float 0..1)."""
    m = mask_u8.astype(np.float32) / 255.0
    return torch.from_numpy(m).unsqueeze(0)


def _face_bboxes_mask(shape_hw, gen_faces, matched_faces):
    """Builds a binary (H,W) float32 mask that is 1.0 inside the
    bounding box of every (face_idx, similarity) pair in matched_faces
    (a union - covers every occurrence of that person, matching what
    annotated_image_N boxes/labels), 0.0 elsewhere."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    for fidx, _ in matched_faces:
        x, y, fw, fh = gen_faces[fidx][0:4]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w, int(round(x + fw)))
        y2 = min(h, int(round(y + fh)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def _process_mask(mask, expansion=0, blur=0.0, threshold=0.0, invert=False):
    """Post-processes a binary/soft mask (2D float32 numpy array, values
    in [0,1]): grows or shrinks it (expansion, in pixels - positive
    dilates, negative erodes), softens its edge (blur, Gaussian radius in
    pixels), optionally re-binarizes it at `threshold` (skipped if
    threshold <= 0, keeping a soft/graduated edge instead - useful
    together with blur for a feathered mask), and optionally inverts it -
    in that order."""
    mask = mask.astype(np.float32)
    if expansion != 0:
        k = abs(int(round(expansion))) * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        if expansion > 0:
            mask = cv2.dilate(mask, kernel)
        else:
            mask = cv2.erode(mask, kernel)
    if blur > 0:
        ksize = int(round(blur))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(1, ksize)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    if threshold > 0:
        mask = (mask >= threshold).astype(np.float32)
    if invert:
        mask = 1.0 - mask
    return np.clip(mask, 0.0, 1.0)


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "The opencv-python (cv2) package is not installed. Install it with: "
            "pip install opencv-python (or opencv-python-headless)."
        )


def _select_face(faces, which_face):
    """Shared 'which face to pick among several detections' logic - used by
    both MykeeFaceDetectAlignCrop and MykeeIdentityCompare."""
    if which_face == "largest":
        return max(faces, key=lambda f: f[2] * f[3])
    if which_face == "leftmost":
        return min(faces, key=lambda f: f[0] + f[2] / 2.0)
    if which_face == "rightmost":
        return max(faces, key=lambda f: f[0] + f[2] / 2.0)
    if which_face == "topmost":
        return min(faces, key=lambda f: f[1] + f[3] / 2.0)
    if which_face == "bottommost":
        return max(faces, key=lambda f: f[1] + f[3] / 2.0)
    return max(faces, key=lambda f: f[-1])  # highest_score


def _filter_small_faces(faces, min_face_size_pct):
    """Filters out detections that are too small relative to the LARGEST
    detected face (and thus likely false positives), BEFORE any which_face/
    position-based selection happens. Size is measured as a percentage of
    the largest detected face's AREA - this is resolution-/composition-
    independent, unlike a threshold relative to the image size, which on
    high-resolution images would fail to filter out detections that are
    obviously tiny compared to the real faces but still large in absolute
    pixel terms."""
    if faces is None or len(faces) <= 1 or min_face_size_pct <= 0:
        return faces
    areas = np.array([f[2] * f[3] for f in faces], dtype=np.float64)
    max_area = areas.max()
    if max_area <= 0:
        return faces
    min_area = max_area * (min_face_size_pct / 100.0)
    keep_mask = areas >= min_area
    if not keep_mask.any():
        return faces
    return faces[keep_mask]


def _detect_faces(bgr, yunet_path, score_threshold, min_face_size_pct=0.0):
    h, w = bgr.shape[:2]
    detector = cv2.FaceDetectorYN_create(yunet_path, "", (w, h), score_threshold, 0.3, 5000)
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    faces = _filter_small_faces(faces, min_face_size_pct)
    if faces is not None and len(faces) > 0:
        # Drop any detection with a non-finite bbox/landmark (NaN/Inf) - a
        # rare but real YuNet false-positive on flat/empty regions (e.g. a
        # masked-out black area). Feeding such a row into
        # estimateAffinePartial2D/warpAffine can produce a degenerate,
        # NaN/Inf-filled "aligned face" that has been observed to crash
        # cv2.dnn's shape inference downstream (NaryEltwiseLayer assertion)
        # instead of just producing a low/garbage similarity score.
        finite_mask = np.isfinite(faces).all(axis=1)
        if not finite_mask.all():
            faces = faces[finite_mask] if finite_mask.any() else faces[:0]
    return faces


_SFACE_RECOGNIZER_CACHE = {}


def _hash_file(path, chunk_size=1024 * 1024):
    """SHA256 of a model file - so we can tell, byte for byte, whether the
    .onnx file the live ComfyUI process resolves to is IDENTICAL to the
    one used in an external diagnostic (e.g. a different file with a
    similar/duplicate name in a second annotators folder, or a partial/
    corrupted download, would show up here as a different hash)."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"<could not hash: {e}>"


def _debug_env_dump(yunet_path, sface_path):
    """Temporary diagnostic dump (see MYKEE-XXXX) - prints everything that
    could plausibly differ between a clean external script and the live
    ComfyUI process at the exact moment MykeeIdentityCompare.run() starts:
    which process/thread this is, the exact cv2 build in use, the exact
    resolved model file paths + their size/hash (to catch a
    different/duplicate/corrupted .onnx being picked up), and GPU memory
    pressure. Safe to remove once the NaryEltwise/Add_44 issue is
    resolved - it only prints, it doesn't change any behavior."""
    import threading
    import os as _os

    print("\n[Mykee DEBUG] ===================================================")
    print(f"[Mykee DEBUG] pid={_os.getpid()} thread={threading.current_thread().name}")
    print(f"[Mykee DEBUG] cv2.__version__={cv2.__version__} cv2.__file__={cv2.__file__}")
    try:
        print(f"[Mykee DEBUG] cv2.getNumThreads()={cv2.getNumThreads()}")
    except Exception as e:
        print(f"[Mykee DEBUG] cv2.getNumThreads() failed: {e}")

    for label, path in (("yunet_path", yunet_path), ("sface_path", sface_path)):
        try:
            size = _os.path.getsize(path)
        except Exception as e:
            size = f"<error: {e}>"
        print(f"[Mykee DEBUG] {label}={path!r} size={size} sha256={_hash_file(path)}")

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            print(
                f"[Mykee DEBUG] torch.cuda: allocated={_torch.cuda.memory_allocated()} "
                f"reserved={_torch.cuda.memory_reserved()} "
                f"free/total={_torch.cuda.mem_get_info()}"
            )
    except Exception as e:
        print(f"[Mykee DEBUG] torch.cuda info failed: {e}")
    print("[Mykee DEBUG] ===================================================\n")


def _get_sface_recognizer(sface_path):
    """Returns a cached cv2.FaceRecognizerSF for this model path, creating
    it once and reusing it afterwards.

    This matters beyond performance: creating a fresh FaceRecognizerSF (and
    therefore a fresh cv2.dnn Net) many times in a tight loop - as
    MykeeIdentityCompare now does, once per detected face, for up to ~10+
    faces in a single run - can trip a shape-inference caching bug in
    OpenCV's ONNX importer (surfaces as a NaryEltwiseLayer
    'findCommonShape' assertion, even though every aligned crop fed in is
    a correctly-shaped 112x112 image). Reusing one recognizer instance per
    model path avoids repeatedly re-importing/re-inferring the graph and
    sidesteps the bug."""
    recognizer = _SFACE_RECOGNIZER_CACHE.get(sface_path)
    if recognizer is None:
        recognizer = cv2.FaceRecognizerSF_create(sface_path, "")
        _SFACE_RECOGNIZER_CACHE[sface_path] = recognizer
    return recognizer


def _embed_face_sface(bgr, face_row, sface_path):
    aligned = MykeeFaceDetectAlignCrop._align_crop(bgr, face_row)
    # Force a fresh, C-contiguous uint8 buffer - a non-contiguous or
    # aliased array feeding into cv2.dnn has also been implicated in the
    # same shape-inference bug mentioned above.
    aligned = np.ascontiguousarray(aligned, dtype=np.uint8)
    recognizer = _get_sface_recognizer(sface_path)
    feature = recognizer.feature(aligned)
    return feature.reshape(-1).astype(np.float32)


def _cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


# ---------------------------------------------------------------------------
# 1) Face detection + alignment + context crop (YuNet)
# ---------------------------------------------------------------------------

class MykeeFaceDetectAlignCrop:
    """
    YuNet face detector: finds the best (or largest) face in the image and
    returns three crops:
      - aligned_face: 112x112, eyes rotated level - this feeds the SFace
                       embedding node (the same normalization SFace's
                       training/inference expects)
      - context_crop: expanded by the given 'crop_factor' and scaled to
                       'crop_size' - this is the reference image to feed
                       into Krea 2 / MiniMax H3
      - ffhq_crop:    NVIDIA's official FFHQ dataset alignment recipe
                       (eye/mouth-based rotation+crop, not a plain resize),
                       scaled to 'ffhq_crop_size' - feed THIS into the
                       StyleGAN Editor node instead of a plain resize; a
                       stylegan2-ffhq-*.pkl generator's own latent space
                       actually represents this framing well, unlike an
                       arbitrary crop (see _ffhq_align_crop's docstring).

    face_mask marks the face region at the original image's resolution
    (feathered); context_crop_face_mask gives the same thing in
    context_crop's own coordinate system/resolution - bind THIS one to a
    node's mask input whose image input is context_crop itself (e.g. Krea
    2's "Reference Latent+" mask_1 socket, if image_1 = context_crop).
    Both masks are ROTATED to match the head's tilt (using the angle
    computed from the two eye landmarks), not a plain axis-aligned bbox -
    so a tilted head won't have the mask spill over or clip real face area.

    mask_offset_x_pct/mask_offset_y_pct correct the CENTER used for both
    context_crop's crop window AND both masks (face_mask and
    context_crop_face_mask) - all three move together, so the mask keeps
    lining up with what context_crop actually shows. aligned_face is
    unaffected (it's built from the 5 landmarks directly, not the bbox
    center, so it doesn't need this correction). Use this when the raw
    detected bbox center isn't quite where you want the face framed
    (glasses, a receding hairline, an unusual angle, etc. can all throw
    off YuNet's bbox a little).

    mask_grow_pct grows/shrinks both mask rectangles around their own
    (possibly offset-corrected) center - independent of crop_factor, which
    only changes context_crop's surrounding context, not the mask's own
    size. Only affects the masks, not context_crop/aligned_face.
    """

    @classmethod
    def INPUT_TYPES(cls):
        yunet_files = _list_annotator_files(".onnx") or ["face_detection_yunet_2023mar.onnx"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (
                    yunet_files,
                    {
                        "default": _default_annotator_choice(yunet_files, "face_detection_yunet_2023mar.onnx"),
                        "tooltip": "YuNet .onnx file placed under models/annotators.",
                    },
                ),
                "crop_factor": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 6.0, "step": 0.1,
                    "tooltip": (
                        "How large context_crop's side is, as a direct multiple of the detected face "
                        "bbox's larger dimension (final_side = max(face_w, face_h) * crop_factor). "
                        "Same convention as Impact-Pack's FaceDetailer 'bbox_crop_factor' widget, so "
                        "the two are directly comparable/interchangeable: crop_factor=1.0 is a tight "
                        "crop with no extra context, 3.0 matches a typical FaceDetailer setting."
                    ),
                }),
                "crop_size": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 8}),
                "which_face": (
                    ["highest_score", "largest", "leftmost", "rightmost", "topmost", "bottommost"],
                    {
                        "default": "highest_score",
                        "tooltip": (
                            "Which face to pick when several are found (e.g. a two-person image). "
                            "'leftmost'/'rightmost': the face on the left/right side of the image "
                            "(by face-center x-coordinate) - useful when the prompt also "
                            "distinguishes people as 'the person on the left', etc."
                        ),
                    },
                ),
                "mask_feather": ("INT", {"default": 15, "min": 0, "max": 200, "step": 1}),
                "mask_offset_x_pct": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                        "tooltip": (
                            "Shifts the CENTER used for context_crop's crop window AND "
                            "both masks left (negative) / right (positive), as a "
                            "percentage of the detected face's own width - e.g. 10 shifts "
                            "everything right by 10% of the face bbox width. aligned_face "
                            "is unaffected (built from landmarks, not the bbox center). "
                            "Use this if the detected bbox is a bit off-center from where "
                            "you actually want the face framed (e.g. glasses, a receding "
                            "hairline, or an unusual angle throwing off YuNet's bbox)."
                        ),
                    },
                ),
                "mask_offset_y_pct": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                        "tooltip": (
                            "Shifts context_crop and both masks up (negative) / down "
                            "(positive), as a percentage of the detected face's own "
                            "height. Same idea as mask_offset_x_pct, just the other axis."
                        ),
                    },
                ),
                "mask_grow_pct": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": -90.0, "max": 300.0, "step": 5.0,
                        "tooltip": (
                            "Grows (positive) or shrinks (negative) BOTH masks around "
                            "their own center, as a percentage of the detected face's "
                            "width/height - independent of crop_factor, which only "
                            "changes context_crop's surrounding context, not the mask "
                            "rectangle's own size. E.g. 20 makes the mask 20% larger on "
                            "each side; -20 makes it 20% smaller. Applied after "
                            "mask_offset_x_pct/mask_offset_y_pct, before mask_feather."
                        ),
                    },
                ),
                "score_threshold": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01}),
                "min_face_size_pct": (
                    "FLOAT",
                    {
                        "default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0,
                        "tooltip": (
                            "Discard detections smaller than this percentage of the LARGEST "
                            "detected face's area, before the which_face selection happens. "
                            "This filters out false-positive, tiny background detections (e.g. a "
                            "patterned bush, fabric print) that would otherwise wrongly get picked "
                            "as the 'most extreme' face. 0 = disabled."
                        ),
                    },
                ),
                "edge_fill": (
                    ["replicate", "reflect", "constant_gray", "constant_black"],
                    {
                        "default": "replicate",
                        "tooltip": (
                            "What to fill the missing strip with if crop_factor would push the square "
                            "crop past the image edge. 'replicate': extends the edge pixel row "
                            "(safe, no duplicated features). 'reflect': mirrors the image - can "
                            "duplicate features (eye, ear) near the face, avoid if the bbox is close "
                            "to the edge. 'constant_gray'/'constant_black': solid-color fill. Also "
                            "used for ffhq_crop's own padding, if any."
                        ),
                    },
                ),
                "ffhq_crop_size": (
                    "INT",
                    {
                        "default": 1024, "min": 128, "max": 2048, "step": 8,
                        "tooltip": (
                            "Output size for ffhq_crop - the same alignment recipe NVIDIA used "
                            "to build the FFHQ dataset (eye/mouth-based rotation+crop, not a "
                            "plain resize), which is what a stylegan2-ffhq-*.pkl generator's "
                            "own latent space actually represents well. Feed this into the "
                            "StyleGAN Editor's image input instead of a plain resize."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("aligned_face", "context_crop", "ffhq_crop", "face_mask", "context_crop_face_mask", "score", "bbox", "num_faces_found")
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    def run(self, image, model_file, crop_factor, crop_size, which_face, mask_feather, score_threshold, edge_fill, min_face_size_pct, mask_offset_x_pct, mask_offset_y_pct, mask_grow_pct, ffhq_crop_size):
        _require_cv2()
        model_path = _resolve_model_path(model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}. Place the file under "
                f"'{_annotators_dir()}'."
            )

        bgr = tensor_to_bgr(image)
        h, w = bgr.shape[:2]

        faces = _detect_faces(bgr, model_path, score_threshold, min_face_size_pct)

        if faces is None or len(faces) == 0:
            raise RuntimeError("No face found in the image (try lowering score_threshold or min_face_size_pct).")

        num_faces_found = len(faces)
        face = _select_face(faces, which_face)

        x, y, fw, fh = face[0:4]
        score = float(face[-1])

        # --- aligned face crop for SFace, using the similarity transform
        # computed from the 5 landmarks ---
        aligned_bgr = self._align_crop(bgr, face)

        # --- padded context crop for diffusion models ---
        # Important: we always crop a true SQUARE region, otherwise the
        # crop_size x crop_size resize would distort the image. If the
        # square would extend past the image edge, the missing part is
        # filled in (per edge_fill) instead of clamping the bbox
        # unevenly (which would give a non-square crop -> distortion).
        #
        # mask_offset_x_pct/mask_offset_y_pct correct the CENTER used for
        # this crop window too, not just the mask rectangles below - if
        # only the mask moved, context_crop would keep pulling from the
        # raw (uncorrected) bbox and the mask would drift away from what
        # the crop actually shows. Shifting the crop window's center by
        # the same amount keeps the corrected face position centered in
        # the crop, the same way the raw bbox center originally was.
        mask_offset_x = fw * (mask_offset_x_pct / 100.0)
        mask_offset_y = fh * (mask_offset_y_pct / 100.0)
        cx, cy = x + fw / 2.0 + mask_offset_x, y + fh / 2.0 + mask_offset_y
        side = int(round(max(fw, fh) * crop_factor))
        side = max(side, 2)
        x0 = int(round(cx - side / 2.0))
        y0 = int(round(cy - side / 2.0))
        x1 = x0 + side
        y1 = y0 + side

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            border_type, border_value = _edge_fill_cv2_params(edge_fill)
            kwargs = {"value": border_value} if border_value is not None else {}
            bgr_padded = cv2.copyMakeBorder(
                bgr, pad_top, pad_bottom, pad_left, pad_right, border_type, **kwargs
            )
            x0 += pad_left
            x1 += pad_left
            y0 += pad_top
            y1 += pad_top
            context = bgr_padded[y0:y1, x0:x1]
        else:
            context = bgr[y0:y1, x0:x1]

        if context.size == 0:
            context = bgr
        scale = crop_size / float(side)
        context = cv2.resize(context, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)

        # --- FFHQ-style aligned crop (the actual preprocessing recipe
        # stylegan2-ffhq-*.pkl generators were trained on) - eye/mouth-
        # based rotation+crop, not a plain resize. Falls back to a resized
        # context_crop on the rare degenerate-landmark case (see
        # _ffhq_align_crop's docstring) rather than failing the whole node
        # over one output.
        ffhq_bgr = self._ffhq_align_crop(bgr, face, ffhq_crop_size, edge_fill)
        if ffhq_bgr is None:
            ffhq_bgr = cv2.resize(context, (ffhq_crop_size, ffhq_crop_size), interpolation=cv2.INTER_LANCZOS4)

        # --- head tilt angle from the two eye landmarks ---
        # (angle_deg = the angle of the bbox's 'w' side relative to
        # horizontal, the same convention cv2.boxPoints expects - if the
        # eyes are level, angle_deg=0, which reproduces the old,
        # axis-aligned behavior)
        right_eye = (float(face[4]), float(face[5]))
        left_eye = (float(face[6]), float(face[7]))
        angle_deg = float(np.degrees(np.arctan2(
            left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]
        )))

        # --- face mask at the ORIGINAL image's resolution, feathered,
        # rotated to match the head's tilt (not an axis-aligned bbox) ---
        # (for regional conditioning, if the downstream node expects the
        # full, uncropped image - e.g. if image_1 were the original image)
        # Centered on the same (offset-corrected) cx/cy as context_crop's
        # own window above, so the mask and the crop move together instead
        # of drifting apart - only mask_grow_pct's size change is unique
        # to the mask itself.
        grow_factor = max(0.01, 1.0 + mask_grow_pct / 100.0)
        mask_w = fw * grow_factor
        mask_h = fh * grow_factor
        mask = np.zeros((h, w), dtype=np.uint8)
        rect = ((cx, cy), (mask_w, mask_h), angle_deg)
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        if mask_feather > 0:
            k = mask_feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # --- face mask in CONTEXT_CROP's own coordinate system/
        # resolution, feathered, with the same tilt angle ---
        # (the crop transform only translates+scales, never rotates, so
        # angle_deg carries over unchanged from the original image to
        # crop space)
        # This is the one that lines up pixel-for-pixel with context_crop,
        # so bind THIS to a node's mask_N input (e.g. Reference Latent+
        # mask_1) whose image_N input is context_crop.
        # (The 'face_mask' above is NOT usable for that - it's in the
        # original, uncropped image's coordinates - different resolution,
        # different offset.)
        # local_x/local_y are the RAW bbox's corner in crop space; adding
        # back mask_offset_*scale re-derives the same offset-corrected
        # center as cx/cy above (x0 already baked the offset into where
        # this crop was taken from, so this naturally lands at the crop's
        # actual center when offset != 0 - not somewhere drifted apart
        # from where context_crop's content actually sits).
        crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        local_x = (x + pad_left - x0) * scale
        local_y = (y + pad_top - y0) * scale
        local_w = fw * scale
        local_h = fh * scale
        local_mask_cx = local_x + local_w / 2.0 + mask_offset_x * scale
        local_mask_cy = local_y + local_h / 2.0 + mask_offset_y * scale
        local_mask_w = local_w * grow_factor
        local_mask_h = local_h * grow_factor
        crop_rect = (
            (local_mask_cx, local_mask_cy),
            (local_mask_w, local_mask_h),
            angle_deg,
        )
        crop_pts = cv2.boxPoints(crop_rect).astype(np.int32)
        cv2.fillPoly(crop_mask, [crop_pts], 255)
        if mask_feather > 0:
            ck = max(1, int(round(mask_feather * scale)))
            ck = ck * 2 + 1
            crop_mask = cv2.GaussianBlur(crop_mask, (ck, ck), 0)

        bbox_str = f"{int(x)},{int(y)},{int(fw)},{int(fh)}"

        return (
            bgr_to_tensor(aligned_bgr),
            bgr_to_tensor(context),
            bgr_to_tensor(ffhq_bgr),
            mask_to_tensor(mask),
            mask_to_tensor(crop_mask),
            score,
            bbox_str,
            num_faces_found,
        )

    @staticmethod
    def _align_crop(bgr, face_row):
        """From the 5 landmarks (right eye, left eye, nose tip, right mouth
        corner, left mouth corner), builds a 112x112 SFace-compatible
        aligned crop, using the same similarity transform that cv2's
        FaceRecognizerSF.alignCrop uses internally."""
        pts_src = np.array(
            [
                [face_row[4], face_row[5]],    # right eye
                [face_row[6], face_row[7]],    # left eye
                [face_row[8], face_row[9]],    # nose tip
                [face_row[10], face_row[11]],  # right mouth corner
                [face_row[12], face_row[13]],  # left mouth corner
            ],
            dtype=np.float32,
        )
        # Target points per the ArcFace/SFace convention, in 112x112 space.
        pts_dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        m, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.LMEDS)
        if m is None or not np.isfinite(m).all():
            x, y, fw, fh = face_row[0:4]
            crop = bgr[int(max(0, y)):int(y + fh), int(max(0, x)):int(x + fw)]
            if crop.size == 0:
                crop = bgr
            return cv2.resize(crop, (112, 112))
        return cv2.warpAffine(bgr, m, (112, 112), borderValue=0.0)

    @staticmethod
    def _ffhq_align_crop(bgr, face_row, output_size, edge_fill):
        """Reimplements NVIDIA's official FFHQ dataset alignment recipe
        (NVlabs/ffhq-dataset's recreate_aligned_images / the widely-mirrored
        align_images.py), the exact preprocessing used to build the FFHQ
        dataset that stylegan2-ffhq-1024x1024.pkl (and similar) were
        trained on - this is what makes a real photo actually look like
        something that generator's own latent space can represent well,
        instead of an arbitrary crop/resize.

        The original recipe averages small landmark clusters for each eye
        (dlib's 68-point model) and reads two specific mouth-outer-corner
        points; YuNet's 5-point output already gives a single point per
        eye and both mouth corners directly, which is what the original
        algorithm's math actually reduces to - so the same geometry
        applies with no extra landmark model needed, just re-pointed at
        YuNet's landmarks.

        Two simplifications vs the NVIDIA original, for a single
        interactive node rather than an offline dataset-building script:
          - No "shrink" pre-downsample step (that's a speed optimization
            for very large wild-caught source images; ComfyUI images are
            already a reasonable size).
          - The final warp is a single cv2 perspective transform at
            output_size directly, not NVIDIA's 4x-supersampled transform
            then downsample - visually very close, just not literally
            pixel-identical to the original dataset's images.
        Returns None if the landmarks are degenerate (e.g. eyes and mouth
        collinear) - the caller falls back to context_crop's own crop in
        that case rather than producing a garbage warp.
        """
        eye_a = np.array([face_row[4], face_row[5]], dtype=np.float64)
        eye_b = np.array([face_row[6], face_row[7]], dtype=np.float64)
        mouth_a = np.array([face_row[10], face_row[11]], dtype=np.float64)
        mouth_b = np.array([face_row[12], face_row[13]], dtype=np.float64)

        eye_avg = (eye_a + eye_b) * 0.5
        eye_to_eye = eye_b - eye_a
        mouth_avg = (mouth_a + mouth_b) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        eye_to_eye_len = float(np.hypot(*eye_to_eye))
        eye_to_mouth_len = float(np.hypot(*eye_to_mouth))
        if eye_to_eye_len < 1e-3 or eye_to_mouth_len < 1e-3:
            return None

        x = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1, 1])
        x_len = float(np.hypot(*x))
        if x_len < 1e-3:
            return None
        x /= x_len
        x *= max(eye_to_eye_len * 2.0, eye_to_mouth_len * 1.8)
        y = np.flipud(x) * np.array([-1, 1])
        c = eye_avg + eye_to_mouth * 0.1
        # PIL QUAD order: upper-left, lower-left, lower-right, upper-right.
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = float(np.hypot(*x)) * 2.0
        if qsize < 1.0:
            return None

        h, w = bgr.shape[:2]
        border = max(int(round(qsize * 0.1)), 3)

        # Crop down to roughly the region of interest first (cheaper than
        # padding the whole original image, and matches the original
        # recipe's own crop-then-pad order).
        crop_x0 = max(int(np.floor(quad[:, 0].min())) - border, 0)
        crop_y0 = max(int(np.floor(quad[:, 1].min())) - border, 0)
        crop_x1 = min(int(np.ceil(quad[:, 0].max())) + border, w)
        crop_y1 = min(int(np.ceil(quad[:, 1].max())) + border, h)
        work = bgr
        if crop_x0 > 0 or crop_y0 > 0 or crop_x1 < w or crop_y1 < h:
            work = bgr[crop_y0:crop_y1, crop_x0:crop_x1]
            quad = quad - np.array([crop_x0, crop_y0])
        if work.size == 0:
            return None

        # Pad if the quad still reaches past this (possibly already
        # cropped) region's edges - same border/fill convention as
        # context_crop, via edge_fill.
        wh, ww = work.shape[:2]
        pad_left = max(int(np.ceil(-quad[:, 0].min())) + border, 0)
        pad_top = max(int(np.ceil(-quad[:, 1].min())) + border, 0)
        pad_right = max(int(np.ceil(quad[:, 0].max() - ww)) + border, 0)
        pad_bottom = max(int(np.ceil(quad[:, 1].max() - wh)) + border, 0)
        if pad_left or pad_top or pad_right or pad_bottom:
            border_type, border_value = _edge_fill_cv2_params(edge_fill)
            kwargs = {"value": border_value} if border_value is not None else {}
            work = cv2.copyMakeBorder(work, pad_top, pad_bottom, pad_left, pad_right, border_type, **kwargs)
            quad = quad + np.array([pad_left, pad_top])

        size = max(8, int(output_size))
        dst = np.array([[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(work, m, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 2) SFace face embedding
# ---------------------------------------------------------------------------

class MykeeFaceEmbedSFace:
    """
    SFace: 128-dim face embedding from MykeeFaceDetectAlignCrop's
    'aligned_face' output. This embedding can be compared with the
    MykeeIdentityScore node to check how similar a face is between two
    images (e.g. reference vs. a generated take).
    """

    @classmethod
    def INPUT_TYPES(cls):
        sface_files = _list_annotator_files(".onnx") or ["face_recognition_sface_2021dec.onnx"]
        return {
            "required": {
                "aligned_face": ("IMAGE", {"tooltip": "112x112 aligned face crop (output of MykeeFaceDetectAlignCrop)."}),
                "model_file": (
                    sface_files,
                    {
                        "default": _default_annotator_choice(sface_files, "face_recognition_sface_2021dec.onnx"),
                        "tooltip": "SFace .onnx file placed under models/annotators.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MYKEE_FACE_EMBEDDING",)
    RETURN_NAMES = ("face_embedding",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    def run(self, aligned_face, model_file):
        _require_cv2()
        model_path = _resolve_model_path(model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SFace model not found: {model_path}. Place the file under "
                f"'{_annotators_dir()}'."
            )
        recognizer = _get_sface_recognizer(model_path)
        bgr = tensor_to_bgr(aligned_face)
        if bgr.shape[0] != 112 or bgr.shape[1] != 112:
            bgr = cv2.resize(bgr, (112, 112))
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        feature = recognizer.feature(bgr)  # (1, 128) float32
        return (feature.reshape(-1).astype(np.float32),)


# ---------------------------------------------------------------------------
# 3) DINOv2 subject embedding (body/clothing)
# ---------------------------------------------------------------------------

class MykeeSubjectEmbedDINOv2:
    """
    DINOv2-base: 768-dim, self-supervised "subject" embedding from the full
    (not just face-cropped) reference image. This is what can estimate
    body-shape/clothing consistency where SFace only ever sees the face -
    e.g. for a user's cloth top/bottom references.
    """

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": (
                    _list_annotator_dirs() or ["dinov2-base"],
                    {"tooltip": "A models/annotators/dinov2-base folder (in HF format), or an HF repo id (e.g. facebook/dinov2-base) to download it."},
                ),
                "pooling": (["cls_token", "mean_patch"], {"default": "cls_token"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("MYKEE_SUBJECT_EMBEDDING",)
    RETURN_NAMES = ("subject_embedding",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    @classmethod
    def _load(cls, model_path, device):
        key = (model_path, device)
        if key in cls._model_cache:
            return cls._model_cache[key]
        from transformers import AutoImageProcessor, AutoModel

        resolved = model_path
        candidate = os.path.join(_annotators_dir(), model_path)
        if os.path.isdir(candidate):
            resolved = candidate

        processor = AutoImageProcessor.from_pretrained(resolved)
        model = AutoModel.from_pretrained(resolved)
        model.eval().to(device)
        cls._model_cache[key] = (processor, model)
        return processor, model

    def run(self, image, model_path, pooling, device):
        from PIL import Image

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        processor, model = self._load(model_path, device)

        arr = image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state  # (1, 1+N, 768)
        if pooling == "cls_token":
            emb = hidden[:, 0, :]
        else:
            emb = hidden[:, 1:, :].mean(dim=1)
        emb = emb.reshape(-1).detach().cpu().numpy().astype(np.float32)
        return (emb,)


# ---------------------------------------------------------------------------
# 4) Identity similarity (cosine) between two embeddings
# ---------------------------------------------------------------------------

class MykeeIdentityScore:
    """
    Cosine similarity between two embeddings (either can be
    MYKEE_FACE_EMBEDDING or MYKEE_SUBJECT_EMBEDDING - the node accepts any
    combination, as long as both inputs have the same dimensionality).
    This lets you - similar to Inline Studio's "score" - automatically
    rank/filter generated takes by how well they preserve the reference
    identity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding_a": (ANY,),
                "embedding_b": (ANY,),
                "pass_threshold": ("FLOAT", {"default": 0.45, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("similarity", "passed")
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    def run(self, embedding_a, embedding_b, pass_threshold):
        a = self._to_embedding_vector(embedding_a, "embedding_a")
        b = self._to_embedding_vector(embedding_b, "embedding_b")
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"The two embeddings have different dimensions ({a.shape[0]} vs {b.shape[0]}) - "
                "only compare embeddings of the same type (both face, or both subject)."
            )
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        sim = float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0
        return (sim, bool(sim >= pass_threshold))

    @staticmethod
    def _to_embedding_vector(value, param_name):
        """Gives a clear error if someone accidentally connects an IMAGE (or
        other non-embedding data) instead of an embedding - without this,
        the node would silently run and produce a meaningless result."""
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3:
                shape = tuple(value.shape)
                raise ValueError(
                    f"'{param_name}' has an image (IMAGE, shape: {shape}) connected to it, "
                    "not an embedding. Identity Score does not take an image - it wants the "
                    "output of the 'Face Embed (SFace)' or 'Subject Embed (DINOv2)' node. "
                    "Correct chain: image -> Face Detect+Align/Crop -> aligned_face -> "
                    "Face Embed (SFace) -> face_embedding -> here."
                )
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value, dtype=np.float32)

        arr = arr.reshape(-1).astype(np.float32)
        if arr.shape[0] not in (128, 768):
            raise ValueError(
                f"'{param_name}' has {arr.shape[0]} elements - this doesn't look like an SFace "
                "(128) or DINOv2 (768) embedding. Check that the Face Embed (SFace) or "
                "Subject Embed (DINOv2) node's output is what's connected here."
            )
        return arr


# ---------------------------------------------------------------------------
# 5) Reference pack: N images -> batch + MiniMax H3 <Picture N> prompt prefix
# ---------------------------------------------------------------------------

MAX_REFERENCES = 9


class MykeeCharacterReferencePack:
    """
    Merges up to 9 reference images (face / body / cloth top / cloth bottom
    / other) into a single IMAGE batch, scaled to a common size, and builds
    the native MiniMax H3 <Picture 1> <Picture 2> ... prompt prefix in the
    order the images are connected - exactly the way Inline Studio's own
    workflow prompt field is built (see the 'prompt' field in the attached
    inline-graph.json).

    The output IMAGE batch can be connected directly to the native
    "MiniMax H3 Reference to Video" node's image-reference input, and
    ref_prompt_prefix can be prepended to the final prompt (with a String
    Concatenate node).
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "target_size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 8}),
        }
        optional = {}
        for i in range(1, MAX_REFERENCES + 1):
            optional[f"image_{i}"] = ("IMAGE",)
            optional[f"role_{i}"] = (
                ["unset", "face", "body", "cloth_top", "cloth_bottom", "other"],
                {"default": "unset"},
            )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("reference_batch", "ref_prompt_prefix", "count")
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    def run(self, target_size, **kwargs):
        images = []
        roles = []
        for i in range(1, MAX_REFERENCES + 1):
            img = kwargs.get(f"image_{i}")
            if img is None:
                continue
            arr = img
            if arr.shape[1] != target_size or arr.shape[2] != target_size:
                arr = torch.nn.functional.interpolate(
                    arr.permute(0, 3, 1, 2),
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            images.append(arr)
            roles.append(kwargs.get(f"role_{i}", "unset"))

        if not images:
            raise RuntimeError("At least one image_N input must be connected.")

        batch = torch.cat(images, dim=0)
        tags = " ".join(f"<Picture {idx}>" for idx in range(1, len(images) + 1))
        role_note = ", ".join(
            f"Picture {idx}={role}" for idx, role in enumerate(roles, start=1) if role != "unset"
        )
        prefix = tags
        if role_note:
            prefix += f"  # {role_note}"

        return (batch, prefix, len(images))


# ---------------------------------------------------------------------------
# 6) All-in-one identity comparison (up to 10 people)
# ---------------------------------------------------------------------------

MAX_PEOPLE = 10


class MykeeIdentityCompare:
    """
    Convenience node: no need to build a separate Face Detect + Face Embed
    chain for every image - this node takes a generated image and up to
    MAX_PEOPLE (10) reference "source" images directly, runs YuNet
    detection and SFace embedding internally, and reports, for each
    reference person, which face in generated_image matches them best.

    - 'num_people' (1-10) is a live switch: it adds/removes the
      source_image_N inputs AND the similarity_N / passed_N /
      annotated_image_N outputs to match, so the node only shows as many
      "slots" as you actually need.
    - EVERY face detected in generated_image is compared against EVERY
      connected source_image_N. A reference person can appear MORE THAN
      ONCE in generated_image (e.g. a side-by-side comparison of several
      takes/generations of the same person) - so every detected face
      that scores at or above pass_threshold against that reference is
      treated as a match, not just the single highest-scoring one.
      similarity_N/passed_N report the single BEST match found (a simple
      pass/fail number), but annotated_image_N boxes/labels EVERY
      matching face for that person, not only the best one. If NOTHING
      passes the threshold for a reference, the closest face is still
      shown (labelled MISMATCH) so the output image isn't blank.
      There's no left/right or position logic anymore (image1_position
      has been removed).
    - A face-detection failure (e.g. no face found in a source_image_N,
      or in generated_image, because score_threshold/min_face_size_pct
      was set too aggressively) no longer aborts the whole node with a
      hard error. Instead it shows a non-blocking ComfyUI toast
      notification (top-right corner, via the standard Toast API) and
      that one person is skipped (similarity_N=-1, passed_N=False) while
      everyone else is still matched normally.
    - annotated_image_N is a copy of generated_image with ONLY that
      reference person's matching face(s) boxed/labelled - one output
      image PER source_image_N input, instead of a single image with
      every match overlaid on top of each other.
    - mask_N is a MASK covering the same matching face(s) as
      annotated_image_N (a union of their bounding boxes) - so it can be
      wired straight into a face-editing/inpainting node without a
      separate detection step. mask_expansion/mask_blur/mask_threshold/
      invert_mask (applied in that order) shape every mask_N output the
      same way: grow/shrink the box(es), soften the edge, optionally
      re-binarize after blurring, optionally invert to select everything
      EXCEPT the matched face area(s). A person with no match (not
      connected, or nothing detected) gets an all-zero mask (all-one if
      invert_mask is on).
    """

    @classmethod
    def INPUT_TYPES(cls):
        yunet_files = _list_annotator_files(".onnx") or ["face_detection_yunet_2023mar.onnx"]
        sface_files = _list_annotator_files(".onnx") or ["face_recognition_sface_2021dec.onnx"]
        required = {
            "generated_image": ("IMAGE",),
            "num_people": (
                "INT",
                {
                    "default": 1, "min": 1, "max": MAX_PEOPLE, "step": 1,
                    "tooltip": (
                        "How many reference people to search for (1-10). Adds/removes the "
                        "source_image_N inputs and similarity_N / passed_N / annotated_image_N "
                        "outputs to match this count."
                    ),
                },
            ),
            "model_file_yunet": (
                yunet_files,
                {"default": _default_annotator_choice(yunet_files, "face_detection_yunet_2023mar.onnx")},
            ),
            "model_file_sface": (
                sface_files,
                {"default": _default_annotator_choice(sface_files, "face_recognition_sface_2021dec.onnx")},
            ),
            "pass_threshold": ("FLOAT", {"default": 0.45, "min": -1.0, "max": 1.0, "step": 0.01}),
            "score_threshold": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01}),
            "min_face_size_pct": (
                "FLOAT",
                {
                    "default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": (
                        "Discard detections smaller than this percentage of the LARGEST "
                        "detected face's area, in BOTH generated_image and every source_image_N "
                        "- this filters out false-positive, tiny background detections. "
                        "0 = disabled."
                    ),
                },
            ),
            "mask_expansion": (
                "INT",
                {
                    "default": 0, "min": -512, "max": 512, "step": 1,
                    "tooltip": (
                        "Grows (positive) or shrinks (negative) every mask_N output, in pixels, "
                        "via morphological dilate/erode. 0 = the raw face bounding box(es), "
                        "unchanged."
                    ),
                },
            ),
            "mask_blur": (
                "FLOAT",
                {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 1.0,
                    "tooltip": (
                        "Softens the edge of every mask_N output with a Gaussian blur (radius in "
                        "pixels, applied AFTER mask_expansion). 0 = a hard edge."
                    ),
                },
            ),
            "mask_threshold": (
                "FLOAT",
                {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Only used when 'mask_blur' > 0. Re-binarizes the blurred mask at this "
                        "cutoff (pixels at/above it become fully opaque, below become fully "
                        "transparent), removing the soft edge blur would otherwise leave. "
                        "0 = disabled, keeping the soft/feathered edge from mask_blur as-is."
                    ),
                },
            ),
            "invert_mask": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Inverts every mask_N output (so everything EXCEPT the matched face area(s) is selected). Applied last, after expansion/blur/threshold.",
                },
            ),
        }
        optional = {}
        for i in range(1, MAX_PEOPLE + 1):
            optional[f"source_image_{i}"] = ("IMAGE",)
        return {"required": required, "optional": optional, "hidden": {"node_id": "UNIQUE_ID"}}

    # 'summary' is listed FIRST (index 0), not last. The web/
    # mykee_identity_compare.js extension adds/removes the per-person
    # similarity_N/passed_N/annotated_image_N/mask_N output SOCKETS as
    # num_people changes, and litegraph's addOutput() always APPENDS new
    # sockets to the end of the outputs array. ComfyUI links reference an
    # output by its numeric slot INDEX, not by name - so if 'summary' sat
    # last and num_people was ever increased, newly-added person outputs
    # would land AFTER it, silently shifting 'summary' out of its
    # original slot and corrupting any link already connected to it
    # (exactly the "received_type(FLOAT) mismatch input_type(STRING)"
    # bug this fixed). Anchoring 'summary' at index 0 means it can never
    # be disturbed by end-appends, regardless of how num_people is
    # changed afterwards.
    RETURN_TYPES = ("STRING",) + tuple(
        t for _ in range(1, MAX_PEOPLE + 1) for t in ("FLOAT", "BOOLEAN", "IMAGE", "MASK")
    )
    RETURN_NAMES = ("summary",) + tuple(
        n
        for i in range(1, MAX_PEOPLE + 1)
        for n in (f"similarity_{i}", f"passed_{i}", f"annotated_image_{i}", f"mask_{i}")
    )
    FUNCTION = "run"
    CATEGORY = "Mykee/Character"

    def run(
        self,
        generated_image,
        num_people,
        model_file_yunet,
        model_file_sface,
        pass_threshold,
        score_threshold,
        min_face_size_pct,
        mask_expansion,
        mask_blur,
        mask_threshold,
        invert_mask,
        node_id=None,
        **kwargs,
    ):
        _require_cv2()
        yunet_path = _resolve_model_path(model_file_yunet)
        sface_path = _resolve_model_path(model_file_sface)
        if not os.path.isfile(yunet_path):
            raise FileNotFoundError(f"YuNet model not found: {yunet_path}.")
        if not os.path.isfile(sface_path):
            raise FileNotFoundError(f"SFace model not found: {sface_path}.")

        _debug_env_dump(yunet_path, sface_path)

        gen_bgr = tensor_to_bgr(generated_image)
        gen_h, gen_w = gen_bgr.shape[:2]

        def _blank_mask():
            m = np.zeros((gen_h, gen_w), dtype=np.float32)
            if invert_mask:
                m = 1.0 - m
            return torch.from_numpy(m).unsqueeze(0)

        def _blank_result(summary_text):
            """Every person gets a neutral (-1.0, False, passthrough
            generated_image, blank mask) result - used when nothing at
            all could be matched, so the node still returns something
            usable instead of aborting the whole prompt. 'summary' goes
            FIRST in the tuple (matching RETURN_TYPES/RETURN_NAMES) - see
            the note by those two attributes for why."""
            out = [summary_text]
            for _ in range(1, MAX_PEOPLE + 1):
                out.extend([-1.0, False, bgr_to_tensor(gen_bgr), _blank_mask()])
            return tuple(out)

        gen_faces = _detect_faces(gen_bgr, yunet_path, score_threshold, min_face_size_pct)
        if gen_faces is None or len(gen_faces) == 0:
            msg = "No face found in generated_image."
            _toast(node_id, "error", "Mykee Identity Compare", msg)
            return _blank_result(msg)
        num_gen_faces = len(gen_faces)

        # Embed every detected face in generated_image once, up front, so
        # every reference person is compared against the same fixed list
        # of embeddings (instead of re-running SFace once per reference).
        # A single bad/false-positive detection (e.g. on a flat masked-out
        # area) must not take down the whole node - if embedding one face
        # fails, it's excluded (gen_embs[idx] = None) and flagged below,
        # instead of the exception propagating out of run().
        gen_embs = []
        lines = [f"{num_gen_faces} face(s) detected in generated_image."]
        for idx, f in enumerate(gen_faces):
            try:
                gen_embs.append(_embed_face_sface(gen_bgr, f, sface_path))
            except cv2.error as e:
                gen_embs.append(None)
                bbox = ",".join(str(int(round(v))) for v in f[0:4])
                lines.append(
                    f"WARNING: face #{idx + 1} in generated_image (bbox {bbox}) could not be "
                    f"embedded and was excluded from matching - likely a bad/false-positive "
                    f"detection. OpenCV error: {e}"
                )

        valid_face_indices = [i for i, ge in enumerate(gen_embs) if ge is not None]
        if not valid_face_indices:
            msg = "None of the faces detected in generated_image could be embedded."
            lines.append(f"WARNING: {msg}")
            _toast(node_id, "error", "Mykee Identity Compare", msg)
            return _blank_result("\n".join(lines))

        matches_by_face = {}  # detected-face index -> [person indices]
        results = {}  # person index -> (best_similarity, best_passed, best_face_idx or None, matched_faces)

        for i in range(1, num_people + 1):
            src = kwargs.get(f"source_image_{i}")
            if src is None:
                lines.append(f"Person {i}: source_image_{i} not connected - skipped.")
                results[i] = (-1.0, False, None, [])
                continue

            src_bgr = tensor_to_bgr(src)
            src_faces = _detect_faces(src_bgr, yunet_path, score_threshold, min_face_size_pct)
            if src_faces is None or len(src_faces) == 0:
                msg = (
                    f"No face found in source_image_{i} (try lowering score_threshold or "
                    "min_face_size_pct) - this person was skipped."
                )
                lines.append(f"WARNING: {msg}")
                _toast(node_id, "warn", "Mykee Identity Compare", msg)
                results[i] = (-1.0, False, None, [])
                continue
            src_face = _select_face(src_faces, "highest_score")
            try:
                src_emb = _embed_face_sface(src_bgr, src_face, sface_path)
            except cv2.error as e:
                msg = (
                    f"Could not compute a face embedding for source_image_{i} - the detected "
                    f"face crop seems to be invalid ({e}) - this person was skipped."
                )
                lines.append(f"WARNING: {msg}")
                _toast(node_id, "warn", "Mykee Identity Compare", msg)
                results[i] = (-1.0, False, None, [])
                continue

            sims = [_cosine_similarity(src_emb, gen_embs[vi]) for vi in valid_face_indices]
            local_best = int(np.argmax(sims))
            best_idx = valid_face_indices[local_best]
            best_sim = float(sims[local_best])
            passed = best_sim >= pass_threshold

            # This reference person can appear MORE THAN ONCE in
            # generated_image (e.g. a side-by-side comparison of several
            # generations/takes of the same person) - so every detected
            # face is checked against the threshold, not just the single
            # best one, and ALL of them get annotated on this person's
            # output image. similarity_i/passed_i still report the single
            # BEST match (for a simple pass/fail number to wire up
            # elsewhere), but annotated_image_i now shows every occurrence.
            matched_faces = [
                (valid_face_indices[j], float(sims[j]))
                for j in range(len(sims))
                if sims[j] >= pass_threshold
            ]
            if not matched_faces:
                # Nothing passed the threshold - still show the closest
                # attempt so the output image isn't just a blank passthrough.
                matched_faces = [(best_idx, best_sim)]
            matched_faces.sort(key=lambda t: t[1], reverse=True)

            results[i] = (best_sim, passed, best_idx, matched_faces)
            for fidx, _ in matched_faces:
                matches_by_face.setdefault(fidx, []).append(i)

            if len(matched_faces) > 1:
                extra = ", ".join(f"face #{fidx + 1} ({fsim:.3f})" for fidx, fsim in matched_faces[1:])
                lines.append(
                    f"Person {i} (source_image_{i}): {len(matched_faces)} face(s) matched above "
                    f"threshold={pass_threshold:.2f} - best = face #{best_idx + 1}, similarity "
                    f"{best_sim:.3f}; also matched {extra}"
                )
            else:
                lines.append(
                    f"Person {i} (source_image_{i}): best match = face #{best_idx + 1} in "
                    f"generated_image, similarity {best_sim:.3f} "
                    f"({'MATCH' if passed else 'MISMATCH'}, threshold={pass_threshold:.2f})"
                )

        for face_idx, people in matches_by_face.items():
            if len(people) > 1:
                names = ", ".join(f"source_image_{p}" for p in people)
                lines.append(
                    f"WARNING: {names} were all matched to the same face "
                    f"(#{face_idx + 1}) in generated_image - they may be the same person, "
                    "or these reference images are too similar for SFace to tell apart."
                )

        summary = "\n".join(lines)

        out = [summary]
        for i in range(1, MAX_PEOPLE + 1):
            sim, passed, face_idx, matched_faces = results.get(i, (-1.0, False, None, []))

            annotated = gen_bgr.copy()
            for fidx, fsim in matched_faces:
                fpassed = fsim >= pass_threshold
                self._draw_face_label(annotated, gen_faces[fidx], f"Person {i}: {fsim:.2f}", fpassed)

            mask = _face_bboxes_mask((gen_h, gen_w), gen_faces, matched_faces)
            mask = _process_mask(mask, mask_expansion, mask_blur, mask_threshold, invert_mask)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            out.extend([float(sim), bool(passed), bgr_to_tensor(annotated), mask_tensor])

        return tuple(out)

    @staticmethod
    def _draw_face_label(bgr, face_row, text, passed):
        """Draws a box around face_row's bbox and writes the text label
        above/below it (with a filled background so it stays readable on
        both dark and light images). Green = meets pass_threshold,
        red = does not."""
        h, w = bgr.shape[:2]
        x, y, fw, fh = [int(round(v)) for v in face_row[0:4]]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        color = (0, 200, 0) if passed else (0, 0, 220)  # BGR: green / red

        thickness = max(2, int(round(min(w, h) / 250)))
        cv2.rectangle(bgr, (x, y), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 900.0)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 2)
        label_y = y - 8 if (y - 8 - th - baseline) > 0 else y2 + th + baseline + 8
        bg_y1 = max(0, label_y - th - baseline - 4)
        bg_y2 = min(h, label_y + baseline + 2)
        bg_x2 = min(w, x + tw + 8)
        cv2.rectangle(bgr, (x, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(bgr, text, (x + 4, label_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

NODE_CLASS_MAPPINGS = {
    "MykeeFaceDetectAlignCrop": MykeeFaceDetectAlignCrop,
    "MykeeFaceEmbedSFace": MykeeFaceEmbedSFace,
    "MykeeSubjectEmbedDINOv2": MykeeSubjectEmbedDINOv2,
    "MykeeIdentityScore": MykeeIdentityScore,
    "MykeeCharacterReferencePack": MykeeCharacterReferencePack,
    "MykeeIdentityCompare": MykeeIdentityCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeFaceDetectAlignCrop": "Mykee Face Detect + Align/Crop (YuNet)",
    "MykeeFaceEmbedSFace": "Mykee Face Embed (SFace)",
    "MykeeSubjectEmbedDINOv2": "Mykee Subject Embed (DINOv2)",
    "MykeeIdentityScore": "Mykee Identity Score (cosine)",
    "MykeeCharacterReferencePack": "Mykee Character Reference Pack",
    "MykeeIdentityCompare": "Mykee Identity Compare (Multi-Person)",
}
