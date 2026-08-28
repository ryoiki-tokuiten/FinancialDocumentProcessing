"""
Terminal UI for autonomous financial document processing agents.

Features:
- Continuous, unbroken global outside box enclosing the entire Parallel Extraction run from start to finish.
- Responsive layout that dynamically adjusts to terminal width and screen sizes.
- True 2-column side-by-side streaming (Gemini Multimodal Python on Left, Sarvam Doc AI on Right).
- Glitch-proof sequential terminal streaming (zero duplicates on scroll or resize).
- Side-by-side parallel extraction comparison summary at completion.
- Transparent syntax highlighting (theme='monokai', background_color='default') with line numbers.
- Zero emojis, high-contrast truecolor ANSI borders (bright blue for thoughts, cyan for actions, bright green for success/verification pass, bold bright red for errors).
- URL-encoded OSC 8 clickable filename hyperlinks (supports paths with spaces).
"""

import os
import re
import json
import urllib.parse
import threading
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.box import Box, ROUNDED
from rich.style import Style

# Responsive console dynamically adapting to current terminal dimensions
console = Console(force_terminal=True, color_system="truecolor")

# Unbroken continuous box components
HEADER_BOX = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "│ ││\n"
)

CONTINUOUS_ROW_BOX = Box(
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
)

FOOTER_BOX = Box(
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "│ ││\n"
    "╰─┴╯\n"
)


def _make_syntax(payload: Any, lang: str = "json") -> Syntax:
    """Helper to generate syntax highlighted code/JSON with native terminal background."""
    if isinstance(payload, (dict, list)):
        code = json.dumps(payload, indent=2)
        lang = "json"
    else:
        code = str(payload).strip()
    return Syntax(code, lang, theme="monokai", background_color="default", line_numbers=True)


class ParallelExtractorUI:
    """
    Manages continuous unbroken global outside box framing for the entire parallel extraction run.
    Streams side-by-side events framed inside the global box and closes it upon completion.
    """
    def __init__(self, cons: Optional[Console] = None):
        self.console = cons or console
        self.lock = threading.Lock()
        self.active = False
        self.gemini_data: Optional[Dict[str, Any]] = None
        self.sarvam_data: Optional[Dict[str, Any]] = None
        self.gemini_passed: Optional[bool] = None
        self.sarvam_passed: Optional[bool] = None

    def start(self, doc_path: str = ""):
        with self.lock:
            self.active = True
            self.gemini_data = None
            self.sarvam_data = None
            self.gemini_passed = None
            self.sarvam_passed = None
            
            doc_name = os.path.basename(doc_path) if doc_path else "document"
            self.console.print("\n")
            
            # Start the continuous Global Outside Box
            header_table = Table(
                title=f"[PARALLEL EXTRACTION ENGINE — {doc_name}]",
                title_style="bold bright_cyan",
                title_justify="center",
                show_header=True,
                header_style="bold bright_cyan",
                border_style="bright_blue",
                box=HEADER_BOX,
                expand=True,
                show_edge=True,
                pad_edge=False
            )
            header_table.add_column("GEMINI AGENT (Multimodal Python)", ratio=1, justify="center")
            header_table.add_column("SARVAM AGENT (Doc AI)", ratio=1, justify="center")
            self.console.print(header_table, end="")

    def _print_row(self, g_panel: Optional[Any] = None, s_panel: Optional[Any] = None):
        # Stream content row framed inside the continuous unbroken side borders and divider
        row_table = Table(
            show_header=False,
            border_style="bright_blue",
            box=CONTINUOUS_ROW_BOX,
            expand=True,
            show_edge=True,
            pad_edge=False,
            padding=(0, 1)
        )
        row_table.add_column(ratio=1)
        row_table.add_column(ratio=1)
        row_table.add_row(g_panel if g_panel is not None else Text(""), s_panel if s_panel is not None else Text(""))
        self.console.print(row_table, end="")

    def stop(self):
        with self.lock:
            if not self.active:
                return
            self.active = False
            
            # Close the continuous Global Outside Box
            footer = Table(
                show_header=False,
                border_style="bright_blue",
                box=FOOTER_BOX,
                expand=True,
                show_edge=True,
                pad_edge=False
            )
            footer.add_column(ratio=1)
            footer.add_column(ratio=1)
            footer.add_row(Text(""), Text(""))
            self.console.print(footer)
            
            # Print side-by-side comparison table if at least one agent extracted data
            if self.gemini_data or self.sarvam_data:
                self._print_comparison_table()
            self.console.print("\n")

    def _print_comparison_table(self):
        t = Table(
            title="[PARALLEL EXTRACTION COMPARISON]",
            box=ROUNDED,
            expand=True,
            header_style="bold bright_cyan"
        )
        t.add_column("Field", style="bold white", width=22)
        t.add_column("GEMINI AGENT (Multimodal Python)", style="cyan", ratio=1)
        t.add_column("SARVAM AGENT (Doc AI)", style="magenta", ratio=1)
        
        g = self.gemini_data or {}
        s = self.sarvam_data or {}
        
        g_vend = g.get("vendor", {}).get("raw_name") or "-"
        s_vend = s.get("vendor", {}).get("raw_name") or "-"
        t.add_row("Vendor Name", str(g_vend), str(s_vend))
        
        g_inv = g.get("invoice_details", {}).get("invoice_number") or "-"
        s_inv = s.get("invoice_details", {}).get("invoice_number") or "-"
        t.add_row("Invoice/Bill Number", str(g_inv), str(s_inv))
        
        g_date = g.get("invoice_details", {}).get("invoice_date") or "-"
        s_date = s.get("invoice_details", {}).get("invoice_date") or "-"
        t.add_row("Invoice Date", str(g_date), str(s_date))
        
        g_sub = g.get("financials", {}).get("subtotal")
        s_sub = s.get("financials", {}).get("subtotal")
        g_sub_str = f"₹{g_sub}" if g_sub is not None else "-"
        s_sub_str = f"₹{s_sub}" if s_sub is not None else "-"
        t.add_row("Subtotal", g_sub_str, s_sub_str)
        
        g_tax = g.get("financials", {}).get("tax_amount")
        s_tax = s.get("financials", {}).get("tax_amount")
        g_tax_str = f"₹{g_tax}" if g_tax is not None else "-"
        s_tax_str = f"₹{s_tax}" if s_tax is not None else "-"
        t.add_row("Tax Amount", g_tax_str, s_tax_str)
        
        g_tot = g.get("financials", {}).get("total_amount")
        s_tot = s.get("financials", {}).get("total_amount")
        g_tot_str = f"₹{g_tot}" if g_tot is not None else "-"
        s_tot_str = f"₹{s_tot}" if s_tot is not None else "-"
        t.add_row("Total Amount", g_tot_str, s_tot_str)
        
        g_stat = "[bold bright_green]PASSED[/bold bright_green]" if self.gemini_passed else ("[bold bright_red]FAILED[/bold bright_red]" if self.gemini_passed is False else "-")
        s_stat = "[bold bright_green]PASSED[/bold bright_green]" if self.sarvam_passed else ("[bold bright_red]FAILED[/bold bright_red]" if self.sarvam_passed is False else "-")
        t.add_row("Verification Status", g_stat, s_stat)
        
        self.console.print(t)

    def add_gemini_thought(self, text: str):
        with self.lock:
            clean = text.strip()
            if not clean or clean.startswith("```json") or clean.startswith("{"):
                return
            
            # Parse LangChain list format if needed
            if clean.startswith("[{'type': 'text'") or clean.startswith('[{"type": "text"'):
                try:
                    import ast
                    parsed = ast.literal_eval(clean)
                    if isinstance(parsed, list) and isinstance(parsed[0], dict) and "text" in parsed[0]:
                        clean = "\n".join(p["text"] for p in parsed if "text" in p).strip()
                except Exception:
                    pass
            
            # Strip trailing Action: ... line
            clean = re.sub(r'\n+\*?\*?Action:\*?\*?\s*[^\n]+.*$', '', clean, flags=re.DOTALL | re.IGNORECASE).strip()
            if not clean:
                return
            
            p = Panel(Markdown(clean, code_theme="monokai"), title="[GEMINI: THOUGHT]", border_style="bright_blue", box=ROUNDED, padding=(0, 1))
            self._print_row(g_panel=p)

    def add_gemini_action(self, tool_name: str, payload: Any):
        with self.lock:
            if tool_name == "execute_python":
                syntax = _make_syntax(payload, "python")
                p = Panel(syntax, title="[GEMINI: ACTION execute_python]", border_style="cyan", box=ROUNDED, padding=(0, 1))
            elif tool_name == "Final_Extraction":
                syntax = _make_syntax(payload, "json")
                if isinstance(payload, dict):
                    self.gemini_data = payload
                p = Panel(syntax, title="[GEMINI: ACTION Final_Extraction]", border_style="cyan", box=ROUNDED, padding=(0, 1))
            else:
                syntax = _make_syntax(payload, "json" if isinstance(payload, (dict, list)) else "python")
                p = Panel(syntax, title=f"[GEMINI: ACTION {tool_name}]", border_style="cyan", box=ROUNDED, padding=(0, 1))
                
            self._print_row(g_panel=p)

    def add_gemini_harness(self, tool_name: str, stdout: str = "", images: Optional[List[str]] = None, success: bool = True, error: str = ""):
        with self.lock:
            body = Text()
            if not success:
                body.append(f"Execution Error:\n{error or 'Unknown error'}\n\n", style="bold bright_red")
            
            if images:
                body.append("Generated crops: ", style="bold bright_white")
                for idx, img in enumerate(images):
                    fname = os.path.basename(img)
                    abs_img = os.path.abspath(img)
                    uri = "file://" + urllib.parse.quote(abs_img)
                    body.append(fname, style=Style(color="bright_cyan", bold=True, underline=True, link=uri))
                    if idx < len(images) - 1:
                        body.append(", ")
                body.append("\n\n")
                
            if stdout:
                clean_stdout = stdout.strip()
                if len(clean_stdout) > 1200:
                    clean_stdout = clean_stdout[:1200] + f"\n... ({len(clean_stdout) - 1200} chars truncated)"
                body.append("STDOUT:\n" + clean_stdout, style="white" if success else "bright_red")
            elif success and not images:
                body.append("Execution completed successfully (no STDOUT).", style="dim")
                
            p = Panel(
                body,
                title=f"[GEMINI: HARNESS {tool_name}]",
                border_style="bright_green" if success else "bold bright_red",
                box=ROUNDED,
                padding=(0, 1)
            )
            self._print_row(g_panel=p)

    def add_gemini_verification(self, passed: bool, errors: Optional[List[str]] = None):
        with self.lock:
            self.gemini_passed = passed
            if passed:
                p = Panel(
                    Text("Mathematical Verification: PASSED", style="bold bright_green"),
                    title="[GEMINI: VERIFICATION PASSED]",
                    border_style="bright_green",
                    box=ROUNDED,
                    padding=(0, 1)
                )
            else:
                err_md = "\n".join([f"- {e}" for e in (errors or ["Math check failed"])])
                p = Panel(
                    Markdown(f"**Verification Failed:**\n{err_md}"),
                    title="[GEMINI: VERIFICATION FAILED]",
                    border_style="bold bright_red",
                    box=ROUNDED,
                    padding=(0, 1)
                )
            self._print_row(g_panel=p)

    def add_sarvam_event(self, text: str, title: str = "[SARVAM: doc_ai]", style: str = "magenta"):
        with self.lock:
            p = Panel(Text(text, style="white"), title=title, border_style=style, box=ROUNDED, padding=(0, 1))
            self._print_row(s_panel=p)

    def add_sarvam_result(self, data: Dict[str, Any], duration: Optional[float] = None):
        with self.lock:
            self.sarvam_data = data
            dur_str = f" ({duration:.1f}s)" if duration else ""
            syntax = _make_syntax(data, "json")
            p = Panel(
                syntax,
                title=f"[SARVAM: Extracted JSON{dur_str}]",
                border_style="magenta",
                box=ROUNDED,
                padding=(0, 1)
            )
            self._print_row(s_panel=p)

    def add_sarvam_verification(self, passed: bool, errors: Optional[List[str]] = None):
        with self.lock:
            self.sarvam_passed = passed
            if passed:
                p = Panel(
                    Text("Mathematical Verification: PASSED", style="bold bright_green"),
                    title="[SARVAM: VERIFICATION PASSED]",
                    border_style="bright_green",
                    box=ROUNDED,
                    padding=(0, 1)
                )
            else:
                err_md = "\n".join([f"- {e}" for e in (errors or ["Math check failed"])])
                p = Panel(
                    Markdown(f"**Verification Notes / Errors:**\n{err_md}"),
                    title="[SARVAM: VERIFICATION FAILED]",
                    border_style="bold bright_red",
                    box=ROUNDED,
                    padding=(0, 1)
                )
            self._print_row(s_panel=p)

_parallel_ui: Optional[ParallelExtractorUI] = None


def get_parallel_ui() -> ParallelExtractorUI:
    global _parallel_ui
    if _parallel_ui is None:
        _parallel_ui = ParallelExtractorUI(console)
    return _parallel_ui


def start_parallel_session(doc_path: str = ""):
    ui = get_parallel_ui()
    ui.start(doc_path)


def end_parallel_session():
    global _parallel_ui
    if _parallel_ui and _parallel_ui.active:
        _parallel_ui.stop()


# -----------------------------------------------------------------------------
# SUPERVISOR UI FUNCTIONS (Emoji-Free, Syntax Highlighted, Raw Transparency)
# -----------------------------------------------------------------------------

def supervisor_thought(thought_text: Any):
    """Render Supervisor internal monologue / reasoning in Markdown."""
    if not thought_text:
        return
    
    clean_text = ""
    if isinstance(thought_text, list):
        extracted = []
        for item in thought_text:
            if isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            elif isinstance(item, str):
                extracted.append(item)
        clean_text = "\n".join(extracted).strip()
    elif isinstance(thought_text, str):
        clean_text = thought_text.strip()
        if clean_text.startswith("[{'type': 'text'") or clean_text.startswith('[{"type": "text"'):
            try:
                import ast
                parsed = ast.literal_eval(clean_text)
                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "text" in parsed[0]:
                    clean_text = "\n".join(p["text"] for p in parsed if "text" in p).strip()
            except Exception:
                pass
    else:
        clean_text = str(thought_text).strip()
        
    if not clean_text:
        return
    
    # Strip any redundant Action: ... line at the end to prevent duplicating supervisor_action
    clean_text = re.sub(r'\n+\*?\*?Action:\*?\*?\s*[^\n]+.*$', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
    if not clean_text:
        return
    
    end_parallel_session()
    
    panel = Panel(
        Markdown(clean_text, code_theme="monokai"),
        title="[SUPERVISOR THOUGHT]",
        border_style="bright_blue",
        box=ROUNDED,
        padding=(0, 2)
    )
    console.print(panel)


def supervisor_action(tool_name: str, tool_args: Dict[str, Any]):
    """Render a Supervisor tool call action panel showing the EXACT raw arguments."""
    end_parallel_session()
    
    syntax = _make_syntax(tool_args or {}, "json")

    panel = Panel(
        syntax,
        title=f"[SUPERVISOR ACTION: {tool_name}]",
        border_style="cyan",
        box=ROUNDED,
        padding=(0, 1)
    )
    console.print(panel)


def supervisor_harness(tool_name: str, raw_result: Any, status: str = "success"):
    """Render harness execution output back to supervisor with raw JSON syntax highlighting."""
    end_parallel_session()
    
    border_color = "bright_green" if status == "success" else "yellow" if status == "warning" else "bold bright_red"
    
    # Check if raw_result is or contains valid JSON
    parsed_json = None
    if isinstance(raw_result, dict):
        parsed_json = raw_result
    elif isinstance(raw_result, str):
        try:
            parsed_json = json.loads(raw_result)
        except Exception:
            parsed_json = None
            
    if parsed_json is not None:
        content = _make_syntax(parsed_json, "json")
    else:
        content = Text(str(raw_result).strip(), style="white" if status == "success" else "bold bright_red")
    
    panel = Panel(
        content,
        title=f"[HARNESS: {tool_name}]",
        border_style=border_color,
        box=ROUNDED,
        padding=(0, 1)
    )
    console.print(panel)


def show_final_summary(result: Dict[str, Any]):
    """Render the clean final processing summary card with clickable OSC 8 hyperlink."""
    end_parallel_session()
    
    doc_path = result.get("document_path", "unknown")
    status = result.get("status", "UNKNOWN")
    doc_type = result.get("doc_type") or "receipt"
    
    raw_score = result.get("fraud_risk_score")
    try:
        fraud_score = float(raw_score) if raw_score is not None else 0.0
    except (ValueError, TypeError):
        fraud_score = 0.0
        
    data = result.get("extracted_data") or {}
    
    filename = os.path.basename(doc_path)
    abs_path = os.path.abspath(doc_path)
    uri = "file://" + urllib.parse.quote(abs_path)
    
    body = Text()
    body.append("Document: ", style="bold white")
    body.append(filename, style=Style(color="bright_cyan", bold=True, underline=True, link=uri))
    body.append("\n")
    
    if status == "COMPLETED":
        body.append("Status: ", style="bold white")
        body.append("COMPLETED (Verified & Committed)\n", style="bold bright_green")
    else:
        body.append("Status: ", style="bold white")
        body.append(f"ERROR - {result.get('message', 'Processing failed')}\n", style="bold bright_red")
        
    body.append("Classification: ", style="bold white")
    body.append(f"{doc_type}\n", style="white")
    
    body.append("Fraud Risk Score: ", style="bold white")
    score_style = "bold bright_green" if fraud_score < 0.4 else "bold bright_red"
    body.append(f"{fraud_score:.2f} ({'CLEAR / LOW RISK' if fraud_score < 0.4 else 'REVIEW REQUIRED'})\n", score_style)
    
    if data and isinstance(data, dict):
        v = data.get("vendor", {}).get("raw_name")
        tot = data.get("financials", {}).get("total_amount")
        inv = data.get("invoice_details", {}).get("invoice_number")
        if v:
            body.append(f"Vendor: {v}\n", style="white")
        if inv:
            body.append(f"Invoice/Bill: {inv}\n", style="white")
        if tot is not None:
            body.append("Total Amount: ", style="bold white")
            body.append(f"₹{tot}\n", style="bold bright_green")
            
    summary_panel = Panel(
        body,
        title="[DOCUMENT PROCESSING SUMMARY]",
        box=ROUNDED,
        border_style="bright_green" if status == "COMPLETED" else "bold bright_red",
        padding=(1, 2)
    )
    console.print("\n")
    console.print(summary_panel)
