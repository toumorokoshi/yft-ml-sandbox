# Assistant at home

This is pet project of mine to create a voice assistant that I can run on local hardware at home.

_NOTE_: this is not the the most practical approach to achieving this. I'm mostly doing this for my own learning and experimentation purposes.

## Constraints and Requirements

- ONNX. All models must use ONNX to allow inference on multiple hardware backends.
- Fully open models. This allows for direct modification to the model as needed.

## Architecture

The overall architecture breaks down the assistant into three components:

1. Speech to text (STT). Converts audio into text.
2. LLM (Natural language understanding and response generation). Converts the text prompt into a response prompt.
3. Text to speech (TTS). Converts the response prompt into audio.
