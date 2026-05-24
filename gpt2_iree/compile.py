import os
import subprocess
import urllib.request
import shutil
import sys
from typing import Sequence
import onnx
import onnx.version_converter
import iree.compiler.tools

# Constants (Rule 5)
DEFAULT_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
DEFAULT_MODEL_NAME = "gpt2-lm-head-10.onnx"
CACHE_DIR = ".cache/gpt2_iree"

def get_workspace_dir() -> str:
    """Helper to get root workspace directory or fallback to current dir."""
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        return os.environ["BUILD_WORKSPACE_DIRECTORY"]
    return os.getcwd()

def get_cache_dir() -> str:
    """Pure helper to get resolved cache directory path."""
    return os.path.join(get_workspace_dir(), CACHE_DIR)

def download_file(url: str, dest_path: str) -> None:
    """IO wrapper: download file from URL (Rule 2)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

def check_file_exists(path: str) -> bool:
    """IO wrapper: checks if file exists."""
    return os.path.exists(path)

def run_command(cmd: Sequence[str]) -> bytes:
    """IO wrapper: runs command (Rule 2)."""
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout

def locate_bazel_runfile(runfile_rel_path: str) -> str | None:
    """IO wrapper: tries to locate a file in Bazel runfiles."""
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        path = os.path.join(runfiles_dir, runfile_rel_path)
        if os.path.exists(path):
            return path
    
    # Walk parent directories to look for external directory
    for prefix in ["", "../", "../../", "../../../"]:
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), prefix, "external", runfile_rel_path))
        if os.path.exists(test_path):
            return test_path
        # Also check without external prefix
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), prefix, runfile_rel_path))
        if os.path.exists(test_path):
            return test_path
            
    return None

def find_import_tool() -> str:
    """Helper to locate the iree-import-onnx tool executable."""
    # 1. Check if it's already in the PATH
    for tool_name in ["iree_import_onnx", "iree-import-onnx"]:
        tool_path = shutil.which(tool_name)
        if tool_path:
            return tool_path

    # 2. Check if we can find it via Bazel runfiles locator
    for candidate in [
        "gpt2_iree/iree_import_onnx",
        "gpt2_iree/iree-import-onnx"
    ]:
        runfile_path = locate_bazel_runfile(candidate)
        if runfile_path:
            return runfile_path

    # 3. Search in Bazel runfiles directory if RUNFILES_DIR is set
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        for root, _, files in os.walk(runfiles_dir):
            for tool_name in ["iree_import_onnx", "iree-import-onnx"]:
                if tool_name in files:
                    return os.path.join(root, tool_name)

    # 4. Search relative to python bin directory
    bin_dir = os.path.dirname(sys.executable)
    for tool_name in ["iree_import_onnx", "iree-import-onnx"]:
        local_path = os.path.join(bin_dir, tool_name)
        if os.path.exists(local_path):
            return local_path

    # 5. Fallback to default command
    return "iree-import-onnx"

def get_model_path(model_url: str = DEFAULT_MODEL_URL, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Coordinates model path discovery and downloading."""
    # Try locating the Bazel http_file dependency target in runfiles
    # Bzlmod's http_file download target name is gpt2_onnx, and filename is 'downloaded'
    for candidate in [
        "gpt2_onnx/file/downloaded",
        "_main~_repo_rules~gpt2_onnx/file/downloaded"
    ]:
        runfile_path = locate_bazel_runfile(candidate)
        if runfile_path:
            return runfile_path

    # Fall back to downloading and caching locally
    cache_path = os.path.join(get_cache_dir(), model_name)
    if not check_file_exists(cache_path):
        print(f"Downloading model {model_name}...")
        download_file(model_url, cache_path)
    return cache_path

def compile_onnx_to_vmfb(onnx_path: str, output_vmfb_name: str) -> str:
    """Compiles ONNX model to IREE VMFB. Returns VMFB path."""
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    mlir_path = os.path.join(cache_dir, f"{output_vmfb_name}.mlir")
    vmfb_path = os.path.join(cache_dir, f"{output_vmfb_name}.vmfb")
    
    # Check if compiled file already exists to avoid re-compilation
    if check_file_exists(vmfb_path):
        return vmfb_path
        
    # Load model and check opset version
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    current_opset = model.opset_import[0].version
    print(f"Model opset version: {current_opset}")
    
    # Convert opset to 17 if it's older
    target_opset = 17
    if current_opset < target_opset:
        print(f"Converting ONNX model from opset {current_opset} to {target_opset}...")
        converted_model = onnx.version_converter.convert_version(model, target_opset)
        onnx_path_converted = os.path.join(cache_dir, f"converted_opset{target_opset}_{DEFAULT_MODEL_NAME}")
        onnx.save(converted_model, onnx_path_converted)
        onnx_path = onnx_path_converted
        print(f"Converted model saved to: {onnx_path}")
        
    # Locate import tool and convert ONNX to MLIR
    import_tool = find_import_tool()
    import_cmd = [import_tool, onnx_path, "-o", mlir_path]
    print(f"Running ONNX import: {' '.join(import_cmd)}")
    run_command(import_cmd)
            
    # Compile MLIR to VMFB
    print(f"Compiling MLIR to VMFB: {mlir_path} -> {vmfb_path}")
    iree.compiler.tools.compile_file(
        mlir_path,
        target_backends=["llvm-cpu"],
        output_file=vmfb_path
    )
    return vmfb_path
