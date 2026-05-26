# Assistant at home

This is pet project of mine to create a voice assistant that I can run on local hardware at home.

_NOTE_: this is not the the most practical approach to achieving this. I'm mostly doing this for my own learning and experimentation purposes.

## Constraints and Requirements

- ONNX. All models must use ONNX to allow inference on multiple hardware backends.
- Fully open models. This allows for direct modification to the model as needed.
- Local execution. This is primarily because it is a pet project to help me better understanding how to perform meaningful optimizations locally.

## Architecture

The overall architecture breaks down the assistant into three components:

1. Speech to text (STT). Converts audio into text.
2. LLM (Natural language understanding and response generation). Converts the text prompt into a response prompt.
3. Text to speech (TTS). Converts the response prompt into audio.

## Design Decisions

### Why ONNX?

- ONNX is a pluggable implementation layer - hopefully, if I want to drop in and replace a component, I just have to have an onnx file rather than replacing it with a new dependency.

## Speech-to-Text (STT) Module Implementation

The STT module is built in Rust using the `ort` crate (ONNX Runtime bindings) and uses the [Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine) model.

### Prerequisites

Make sure you have Bazel installed on your system.

### Setup and Model Download

To download all required model assets (tokenizer and ONNX files for both Moonshine STT and Qwen3 LLM) directly from Hugging Face:

```bash
bazel run //assistant_at_home:assistant_at_home -- download
```

This places the model files in `assistant_at_home/models/`.

### Run Transcription

To transcribe a 16kHz mono WAV file to text:

```bash
bazel run -c opt //assistant_at_home:assistant_at_home -- transcribe <PATH_TO_WAV>
```

### Run End-to-End Voice Assistant Pipeline

To run the full end-to-end voice assistant flow (which transcribes a WAV file and feeds the output directly to the Qwen3 LLM):

```bash
bazel run -c opt //assistant_at_home:assistant_at_home -- pipeline <PATH_TO_WAV>
```

For example, to test using the sample Beckett audio:
```bash
# Download sample wav
curl -L -o assistant_at_home/beckett.wav https://raw.githubusercontent.com/moonshine-ai/moonshine-tflite/main/assets/beckett.wav

# Run pipeline
bazel run -c opt //assistant_at_home:assistant_at_home -- pipeline assistant_at_home/beckett.wav
```

### Running Tests

We strictly separate our codebase into pure logic functions operating on data structures (tensors, caches) and I/O wrapper functions (file/network loading) to make testing straightforward.

* **Run Pure Unit Tests** (no I/O or network requests):
  ```bash
  bazel test //assistant_at_home:assistant_at_home_test
  ```

* **Run Integration Tests** (requires downloaded models and test WAV):
  ```bash
  bazel test //assistant_at_home:assistant_at_home_test --test_arg=--ignored --test_output=all
  ```
