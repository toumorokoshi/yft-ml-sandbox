import iree.runtime as ireert
import numpy as np

# Constants (Rule 5)
DEFAULT_DRIVER = "local-task"

def load_iree_context(vmfb_path: str, driver: str = DEFAULT_DRIVER) -> ireert.system_api.BoundModule:
    """IO wrapper: loads compiled VMFB as a BoundModule (Rule 2)."""
    # Load the flatbuffer file using IREE's mmap-friendly file loader.
    # This returns a BoundModule instance with callable methods for each exported function.
    vm_module = ireert.load_vm_flatbuffer_file(vmfb_path, driver=driver)
    return vm_module

def run_iree_inference(module: ireert.system_api.BoundModule, input_ids: np.ndarray) -> np.ndarray:
    """IO wrapper: runs inference on the compiled BoundModule (Rule 2)."""
    # Dynamically resolve the model's entry point function name
    func_name = None
    available_funcs = []
    try:
        available_funcs = list(module.vm_module.function_names)
    except Exception:
        pass
        
    for candidate in ["torch-jit-export", "main", "forward"]:
        if candidate in available_funcs:
            func_name = candidate
            break
            
    if func_name is None:
        # Fallback to the first non-initialization function
        for f in available_funcs:
            if f != "__init" and not f.endswith("$async"):
                func_name = f
                break
                
    if func_name is None:
        raise RuntimeError(f"Could not find any callable entrypoint function in IREE module. Available: {available_funcs}")

    # Invoke the function with input_ids
    func = getattr(module, func_name)

    # Invoke the function with input_ids
    result = func(input_ids)
    
    # ONNX Zoo gpt2-lm-head-10.onnx outputs a tuple (prediction_scores, past)
    # The first element contains the logits
    if isinstance(result, (list, tuple)):
        logits = result[0]
    else:
        logits = result
        
    # If the returned value is an IREE DeviceArray, copy it to host memory as a numpy array
    if hasattr(logits, "to_host"):
        return logits.to_host()
    return np.array(logits)

def greedy_predict_next_token(logits: np.ndarray) -> int:
    """Pure function: takes prediction logits and returns the next token ID (greedy search)."""
    # logits shape: (batch_size, sequence_length, vocab_size)
    # Get the logits for the last token in the sequence
    last_token_logits = logits[0, -1, :]
    return int(np.argmax(last_token_logits))
