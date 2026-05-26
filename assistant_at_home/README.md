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

Make sure you have Rust/Cargo installed, and that your system is configured to link against the ONNX Runtime library (the `ort` crate automatically downloads/links this during build).

### Setup and Model Download

To download the required Moonshine model assets (tokenizer and ONNX encoder/decoder files) directly from Hugging Face:

```bash
cargo run -- download
```

This places the model files in `models/tokenizer.json` and `models/tiny/`.

### Run Transcription

To transcribe a 16kHz mono WAV file to text:

```bash
cargo run --release -- transcribe <PATH_TO_WAV>
```

For example, to test using the sample Beckett audio:
```bash
# Download sample wav
curl -L -o beckett.wav https://raw.githubusercontent.com/moonshine-ai/moonshine-tflite/main/assets/beckett.wav

# Transcribe
cargo run --release -- transcribe beckett.wav
```

### Running Tests

We strictly separate our codebase into pure logic functions operating on data structures (tensors, caches) and I/O wrapper functions (file/network loading) to make testing straightforward.

* **Run Pure Unit Tests** (no I/O or network requests):
  ```bash
  cargo test
  ```

* **Run Integration Test** (requires downloaded models and test WAV):
  ```bash
  cargo test -- --ignored
  ```
