from .calculator import Calculator
from .read_file import ReadFileTool
from .write_file import WriteFileTool
from .glob import GlobTool
from .grep import GrepTool

try:
    from .web_search import WebSearchTool
    _has_web_search = True
except ImportError:
    _has_web_search = False

try:
    from .python_repl import PythonREPLTool
    _has_python_repl = True
except ImportError:
    _has_python_repl = False

try:
    from .http_request import HTTPRequestTool
    _has_http_request = True
except ImportError:
    _has_http_request = False

# ROI tools (optional import)
try:
    from .roi import RoiTool, RoiBatchTool
    _has_roi = True
except Exception:
    _has_roi = False

__all__ = ["Calculator", "ReadFileTool", "WriteFileTool", "GlobTool", "GrepTool"]

if _has_web_search:
    __all__.append("WebSearchTool")
if _has_python_repl:
    __all__.append("PythonREPLTool")
if _has_http_request:
    __all__.append("HTTPRequestTool")
if _has_roi:
    __all__.extend(["RoiTool", "RoiBatchTool"])
