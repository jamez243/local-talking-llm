import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService
import torch

console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

# Print Bark device info and test TTS speed
console.print(f"[bold blue]Bark TTS device: {tts.device}")
if tts.device == "cpu":
    console.print("[bold red]WARNING: Bark is running on CPU. TTS will be extremely slow. Consider installing CUDA drivers and PyTorch with GPU support.")
else:
    console.print("[bold green]Bark is running on GPU. TTS should be much faster.")

# Test TTS with a short string at startup
import time as _time
console.print("[yellow]Testing Bark TTS with 'Hello world'...")
_t0 = _time.time()
try:
    _sr, _arr = tts.long_form_synthesize("Hello world")
    _dt = _time.time() - _t0
    console.print(f"[green]Startup TTS test took {_dt:.2f} seconds.")
except Exception as e:
    console.print(f"[red]Startup TTS test failed: {e}")

template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(),
)


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=True)  # Set fp16=True to use GPU if available
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            console.print(f"[green]Audio numpy array size: {audio_np.size}")
            # Debug: print shape and dtype
            console.print(f"[green]Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

            if audio_np.size > 0:
                start_time = time.time()
                try:
                    with console.status("Transcribing...", spinner="earth"):
                        text = transcribe(audio_np)
                except Exception as e:
                    console.print(f"[red]Transcription error: {e}")
                    continue
                transcribe_time = time.time() - start_time
                console.print(f"[yellow]You: {text}")
                console.print(f"[magenta]Transcription took {transcribe_time:.2f} seconds.")

                start_time = time.time()
                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                llm_time = time.time() - start_time
                console.print(f"[magenta]LLM response took {llm_time:.2f} seconds.")

                start_time = time.time()
                sample_rate, audio_array = tts.long_form_synthesize(response)
                tts_time = time.time() - start_time
                console.print(f"[magenta]TTS synthesis took {tts_time:.2f} seconds.")

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
