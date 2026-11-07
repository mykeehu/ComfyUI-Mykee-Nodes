"""
Shared path guard for the REST routes of the template nodes.

The HTTP routes (/mykee/model_templates/*, /mykee/prompt_templates/*) accept a
"path" parameter from the browser. ComfyUI has no authentication, so anybody
who can reach the server can send any value there. Before this guard existed
that value was used as-is (and even created with os.makedirs), which allowed
listing arbitrary folders and creating / overwriting <anything>.json anywhere
the ComfyUI process can write.

Now a custom folder is only accepted if, after resolving symlinks and "..",
it lies inside one of the allowed roots:

  * ComfyUI's user, input and output folders
  * every folder listed in  allowed_template_dirs.txt  (pack root, one path
    per line, lines starting with # are comments)
  * every folder in the MYKEE_TEMPLATE_ROOTS environment variable
    (separated with ";" on Windows, ":" on Linux/macOS)

Anything else raises PermissionError, which the routes turn into HTTP 403.
"""

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALLOW_FILE = os.path.join(_PACK_ROOT, "allowed_template_dirs.txt")


def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(p))))


def allowed_roots():
    roots = []
    if folder_paths is not None:
        for getter in ("get_user_directory", "get_input_directory", "get_output_directory"):
            fn = getattr(folder_paths, getter, None)
            if callable(fn):
                try:
                    d = fn()
                    if d:
                        roots.append(d)
                except Exception:
                    pass
    try:
        with open(_ALLOW_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip().strip('"')
                if line and not line.startswith("#"):
                    roots.append(line)
    except OSError:
        pass
    for part in os.environ.get("MYKEE_TEMPLATE_ROOTS", "").split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            roots.append(part)
    return [_real(r) for r in roots]


def check_custom_dir(custom_path):
    """Returns the resolved folder if it is inside an allowed root, otherwise
    raises PermissionError. Never creates anything."""
    if "\x00" in custom_path:
        raise PermissionError("invalid path")
    real = _real(custom_path)
    for root in allowed_roots():
        try:
            if os.path.commonpath([real, root]) == root:
                return real
        except ValueError:  # different drives
            continue
    raise PermissionError(
        "This folder is not allowed. Put it under ComfyUI's user/input/output "
        "folder, or list it in allowed_template_dirs.txt in the "
        "ComfyUI-Mykee-Nodes folder (one path per line)."
    )
