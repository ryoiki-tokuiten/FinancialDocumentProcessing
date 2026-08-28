"""
Extractor-specific tools for document analysis with Python code execution.

Contains:
1. execute_python - Run Python code for image analysis with auto-install
2. Final_Extraction - Submit final extraction with inline verification
"""

import json
import logging
import subprocess
import base64
import os
import re
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Allowed image extensions for result capture
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}


def _install_package(package_name: str) -> tuple[bool, str]:
    """
    Install a Python package using pip.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        Tuple of (success, message)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return True, f"Successfully installed {package_name}"
        return False, f"Failed to install {package_name}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout installing {package_name}"
    except Exception as e:
        return False, f"Error installing {package_name}: {str(e)}"


def _parse_import_error(error_message: str) -> Optional[str]:
    """
    Parse an ImportError/ModuleNotFoundError to extract the missing module name.
    
    Args:
        error_message: The error message from stderr
        
    Returns:
        Module name if found, None otherwise
    """
    patterns = [
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        r"ImportError: No module named ['\"]([^'\"]+)['\"]",
        r"No module named ['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_message)
        if match:
            module = match.group(1).split('.')[0]  # Get top-level module
            return module
    return None


def _get_pip_package_name(module_name: str) -> str:
    """
    Map common module names to their pip package names.
    
    Args:
        module_name: The Python module name
        
    Returns:
        The pip package name
    """
    module_to_pip = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'yaml': 'pyyaml',
    }
    return module_to_pip.get(module_name, module_name)


def _encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def _find_generated_images(work_dir: str, before_files: set) -> List[Dict[str, Any]]:
    """
    Find images generated directly in the backend outputs directory.
    """
    images = []
    try:
        for root, _, files in os.walk(work_dir):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                ext = Path(filename).suffix.lower()
                
                if ext in IMAGE_EXTENSIONS and filepath not in before_files:
                    encoded = _encode_image_to_base64(filepath)
                    if encoded:
                        images.append({
                            "filename": filename,
                            "filepath": str(Path(filepath).absolute()),
                            "mime_type": f"image/{ext[1:]}",
                            "base64": encoded
                        })
    except Exception as e:
        logger.error(f"Error finding generated images: {e}")
    
    return images


from .token_limiter import truncate_tool_output
from .forensics import forensics

class StatefulPythonSession:
    """
    Manages a persistent, stateful in-memory Python environment across agent turns.
    Executes directly in backend/outputs so all created images are immediately web-accessible.
    """
    def __init__(self, session_key: str, doc_path: str):
        self.session_key = session_key
        self.doc_path = doc_path
        self.work_dir = str((Path(__file__).parent / "outputs").absolute())
        os.makedirs(self.work_dir, exist_ok=True)
        self.known_files = set()
        self._update_known_files()
        self.session_globals = {}
        self.session_locals = {}
        self.init_environment()

    def _update_known_files(self):
        self.known_files = set()
        for root, _, files in os.walk(self.work_dir):
            for f in files:
                self.known_files.add(os.path.join(root, f))

    def init_environment(self):
        """Initializes the persistent namespace with pre-warmed libraries and utilities."""
        import pathlib
        self.session_globals = {
            "__name__": "__main__",
            "os": os,
            "sys": sys,
            "re": re,
            "json": json,
            "Path": pathlib.Path,
            "forensics": forensics,
            "DOCUMENT_PATH": self.doc_path,
            "doc_path": self.doc_path,
        }
        
        # Pre-warm common libraries safely
        for mod_name, alias in [("cv2", "cv2"), ("numpy", "np"), ("pandas", "pd"), 
                                ("pdfplumber", "pdfplumber"), ("pypdf", "pypdf"), ("easyocr", "easyocr")]:
            try:
                mod = __import__(mod_name)
                self.session_globals[alias] = mod
                self.session_globals[mod_name] = mod
            except Exception:
                pass
                
        try:
            from PIL import Image
            self.session_globals["Image"] = Image
        except Exception:
            pass

        self.session_locals = {}

    def reset(self):
        """Resets the persistent execution state and cleans working directory."""
        self.session_locals.clear()
        self.init_environment()
        try:
            for root, _, files in os.walk(self.work_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
        except Exception:
            pass
        self._update_known_files()

    def execute(self, code: str, view_images: Optional[List[str]] = None) -> Dict[str, Any]:
        """Executes code in the persistent namespace and captures stdout/stderr/images."""
        import io
        import contextlib
        self._update_known_files()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        installed_packages = []
        old_cwd = os.getcwd()

        try:
            os.chdir(self.work_dir)
            os.environ["DOCUMENT_PATH"] = self.doc_path
            
            # Execute with auto-install retry for missing modules
            max_install_attempts = 3
            exec_success = False
            exec_error = ""

            for attempt in range(max_install_attempts + 1):
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                        exec(code, self.session_globals, self.session_locals)
                    exec_success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    missing_mod = _parse_import_error(err_msg)
                    if missing_mod and attempt < max_install_attempts:
                        pip_pkg = _get_pip_package_name(missing_mod)
                        logger.info(f"Auto-installing missing module: {pip_pkg}")
                        success, _ = _install_package(pip_pkg)
                        if success:
                            installed_packages.append(pip_pkg)
                            continue
                    exec_error = f"{type(e).__name__}: {err_msg}"
                    break

            raw_stdout = stdout_capture.getvalue()
            raw_stderr = stderr_capture.getvalue()
            
            # Find generated images in work_dir
            generated_images = _find_generated_images(self.work_dir, self.known_files)
            
            # Handle specific view_images requested by model
            explicit_images = []
            if view_images and isinstance(view_images, list):
                persist_dir = Path(__file__).parent / "outputs"
                uploads_dir = Path(__file__).parent / "uploads"
                for img_name in view_images:
                    if not img_name or not isinstance(img_name, str):
                        continue
                    clean_name = os.path.basename(img_name.strip())
                    candidates = [
                        Path(self.work_dir) / clean_name,
                        Path(self.work_dir) / img_name,
                        persist_dir / clean_name,
                        uploads_dir / clean_name,
                        Path(img_name)
                    ]
                    for cand in candidates:
                        if cand.exists() and cand.is_file():
                            b64 = _encode_image_to_base64(str(cand))
                            if b64:
                                ext = cand.suffix.lower().lstrip(".") or "png"
                                explicit_images.append({
                                    "filename": cand.name,
                                    "filepath": str(cand.absolute()),
                                    "mime_type": f"image/{ext}",
                                    "base64": b64
                                })
                                break

            # Merge and limit images to max 5 per turn
            all_images = []
            seen_files = set()
            for img in (explicit_images + generated_images):
                fp = img.get("filepath", "")
                if fp not in seen_files:
                    seen_files.add(fp)
                    all_images.append(img)
                    
            capped_images = all_images[:5]

            # Apply 5000-token limit to stdout and stderr
            clean_stdout = truncate_tool_output(raw_stdout)
            clean_stderr = truncate_tool_output(raw_stderr)

            return {
                "success": exec_success,
                "stdout": clean_stdout,
                "stderr": clean_stderr,
                "images": capped_images,
                "installed_packages": installed_packages,
                "error": exec_error if not exec_success else ""
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": truncate_tool_output(stdout_capture.getvalue()),
                "stderr": truncate_tool_output(stderr_capture.getvalue()),
                "images": [],
                "error": str(e)
            }
        finally:
            os.chdir(old_cwd)


# Global Session Registry per Document Path
_PERSISTENT_SESSIONS: Dict[str, StatefulPythonSession] = {}

def get_or_create_session(doc_path: str) -> StatefulPythonSession:
    import hashlib
    session_key = hashlib.md5(doc_path.encode()).hexdigest()[:10] if doc_path else "default"
    if session_key not in _PERSISTENT_SESSIONS:
        _PERSISTENT_SESSIONS[session_key] = StatefulPythonSession(session_key, doc_path)
    return _PERSISTENT_SESSIONS[session_key]


@tool
def execute_python(
    code: str, 
    view_images: Optional[List[str]] = None, 
    reset_state: bool = False
) -> Dict[str, Any]:
    """
    Execute Python code in a persistent, stateful analysis session for financial document forensics.
    
    KEY CAPABILITIES:
    - STATEFUL: Variables (e.g. `img`, `df`, `crops`), imported modules, and objects persist across turns.
    - PRE-WARMED LIBS: `cv2`, `np`, `pd`, `Image` (PIL), `pdfplumber`, `pypdf`, `easyocr`, `forensics`.
    - FORENSIC TOOLKIT:
      * `forensics.deskew(img)`: Rotates and straightens tilted invoices.
      * `forensics.ela(img)`: Error Level Analysis to detect pixel-level digital tampering/editing.
      * `forensics.extract_table_grid(img)`: Extracts horizontal/vertical table line structures.
      * `forensics.normalize_currency(str)`: Normalizes non-decimal UK £/s/d, Indian Lakhs/Crores, EUR commas.
    - MULTI-MODAL INSPECTION: Pass `view_images=['crop1.png']` to attach up to 5 visual crops as multi-modal vision input.
    - STATE RESET: Set `reset_state=True` to clear all variables and reset environment when needed.
    
    Args:
        code: Python code to execute.
        view_images: Optional list of saved image filenames (e.g. ['crop_total.png']) to view visually.
        reset_state: Set to True if you want to clear variables and reset the Python environment.
        
    Returns:
        Dictionary with execution success, stdout, stderr, images, and any error message.
    """
    doc_path = os.environ.get("CURRENT_DOCUMENT_PATH", "")
    session = get_or_create_session(doc_path)
    
    if reset_state:
        logger.info(f"Resetting stateful Python environment for session {session.session_key}")
        session.reset()
        if not code or not code.strip():
            return {
                "success": True,
                "stdout": "Environment and variables successfully reset to initial clean state.",
                "stderr": "",
                "images": [],
                "installed_packages": []
            }
            
    return session.execute(code, view_images=view_images)


def run_verification(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates extracted financial data against real-world financial integrity rules.
    
    Supports:
    1. Standard B2B Invoices (Tax-Exclusive): sum(line_items) ≈ subtotal, and subtotal + tax ≈ total
    2. Retail / B2C Receipts (Tax-Inclusive / MRP): sum(line_items) ≈ total, and subtotal + tax ≈ total
    3. Cash Round-Off adjustments (e.g. ±1.00)
    4. Floating point decimal tolerance (0.05 / 1.05 for cash rounding)
    
    Args:
        data: The extracted JSON data
        
    Returns:
        Dict with status (PASS/FAIL) and errors list
    """
    errors = []
    
    if not data:
        return {"status": "FAIL", "errors": ["No data provided"]}
    
    financials = data.get("financials", {})
    line_items = data.get("line_items", [])
    
    # 1. Total amount is required
    total_amount = financials.get("total_amount")
    total_amount_val = None
    if total_amount is None:
        errors.append("Missing required field: financials.total_amount")
    else:
        try:
            total_amount_val = float(total_amount)
        except (ValueError, TypeError):
            errors.append(f"Invalid total_amount: {total_amount}")
    
    # 2. Check vendor name is present
    vendor = data.get("vendor", {})
    if not vendor.get("raw_name"):
        errors.append("Missing required field: vendor.raw_name")
        
    # 3. Calculate sum of line item row_totals
    calculated_line_items_sum = 0.0
    valid_line_items = True
    for idx, item in enumerate(line_items):
        row_total = item.get("row_total")
        if row_total is None:
            errors.append(f"Line item {idx + 1} missing row_total")
            valid_line_items = False
        else:
            try:
                calculated_line_items_sum += float(row_total)
            except (ValueError, TypeError):
                errors.append(f"Line item {idx + 1} has invalid row_total: {row_total}")
                valid_line_items = False
    
    calculated_line_items_sum = round(calculated_line_items_sum, 2)
    
    # 4. Verify line_items_sum matches calculated sum (if provided)
    line_items_sum = financials.get("line_items_sum")
    if line_items_sum is not None and valid_line_items and line_items:
        try:
            rep_sum = float(line_items_sum)
            if abs(rep_sum - calculated_line_items_sum) > 0.05:
                errors.append(
                    f"Line items sum mismatch: reported {line_items_sum}, "
                    f"calculated {calculated_line_items_sum:.2f}"
                )
        except (ValueError, TypeError):
            errors.append(f"Invalid line_items_sum value: {line_items_sum}")
    
    # 5. Financial Math: Subtotal + Tax = Total (with ±1.05 round-off tolerance)
    subtotal = financials.get("subtotal")
    tax_amount = financials.get("tax_amount")
    subtotal_val = None
    tax_val = 0.0
    
    if subtotal is not None:
        try:
            subtotal_val = float(subtotal)
        except (ValueError, TypeError):
            errors.append(f"Invalid subtotal value: {subtotal}")
            
    if tax_amount is not None:
        try:
            tax_val = float(tax_amount)
        except (ValueError, TypeError):
            errors.append(f"Invalid tax_amount value: {tax_amount}")
            
    if subtotal_val is not None and total_amount_val is not None:
        expected_total = round(subtotal_val + tax_val, 2)
        diff_to_total = abs(expected_total - total_amount_val)
        # Allow up to 1.05 for cash rounding adjustments in retail receipts
        if diff_to_total > 1.05:
            errors.append(
                f"Math error: subtotal ({subtotal_val:.2f}) + tax ({tax_val:.2f}) = "
                f"{expected_total:.2f}, but total is {total_amount_val:.2f} (diff: {diff_to_total:.2f})"
            )
            
    # 6. Line Items Consistency: Check B2B (tax-exclusive) OR Retail (tax-inclusive)
    if line_items and valid_line_items:
        is_tax_exclusive_match = False
        is_tax_inclusive_match = False
        
        if subtotal_val is not None:
            # Matches subtotal (B2B Tax-Exclusive)
            if abs(calculated_line_items_sum - subtotal_val) <= 0.10:
                is_tax_exclusive_match = True
                
        if total_amount_val is not None:
            # Matches total (Retail Tax-Inclusive / MRP)
            if abs(calculated_line_items_sum - total_amount_val) <= 1.05:
                is_tax_inclusive_match = True
                
        if subtotal_val is not None and tax_val > 0:
            # Matches subtotal + tax (Retail gross sum)
            if abs(calculated_line_items_sum - (subtotal_val + tax_val)) <= 1.05:
                is_tax_inclusive_match = True
                
        # If both subtotal and total are provided, line items must match at least one mode
        if (subtotal_val is not None or total_amount_val is not None) and not (is_tax_exclusive_match or is_tax_inclusive_match):
            errors.append(
                f"Line items inconsistency: line items sum to {calculated_line_items_sum:.2f}, "
                f"which does not match Subtotal ({subtotal_val}) in B2B mode "
                f"nor Total ({total_amount_val}) in Retail tax-inclusive mode."
            )
    
    if errors:
        return {"status": "FAIL", "errors": errors}
    
    return {"status": "PASS", "errors": []}


@tool
def Final_Extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit the final extracted data for verification.
    
    Call this tool when you are confident in your extraction. The system will
    automatically verify mathematical consistency (subtotal + tax = total, etc).
    
    If verification fails, you will receive specific error messages to fix.
    Continue iterating until verification passes.
    
    Args:
        data: The extracted JSON data matching the schema:
              {
                "vendor": {"raw_name": str, "address": str|null, "tax_id": str|null},
                "invoice_details": {...},
                "financials": {"subtotal": num, "tax_amount": num, "total_amount": num, ...},
                "line_items": [{"description": str, "row_total": num, ...}, ...]
              }
    
    Returns:
        Dictionary with:
        - status: "PASS" or "FAIL"
        - errors: List of error messages (empty if PASS)
        - data: The submitted data (if PASS)
    
    Example success:
        {"status": "PASS", "errors": [], "data": {...}}
    
    Example failure:
        {"status": "FAIL", "errors": ["Math error: subtotal (100) + tax (8) = 108, but total is 110"]}
    """
    logger.info("Final_Extraction called, running verification...")
    
    result = run_verification(data)
    
    if result["status"] == "PASS":
        logger.info("Extraction verified successfully")
        return {
            "status": "PASS",
            "errors": [],
            "data": data
        }
    else:
        logger.info(f"Extraction verification failed: {result['errors']}")
        return {
            "status": "FAIL",
            "errors": result["errors"]
        }
