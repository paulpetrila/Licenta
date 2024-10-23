import whisper
import sounddevice as sd
import numpy as np
import time

# Load the Whisper model
model = whisper.load_model("base")


def listen(duration=5, log_file="logs.txt", temp_file="temp.txt"):
    """
    Listens to audio from mic, writing it in a temp and log file.

    :param duration: how many seconds per listening interval
    :type duration: int
    :param log_file: path to the log of the whole transcript, each session begins with a unix timestamp
    :type log_file: str
    :param temp_file: where to 
    """

    print("Listening...")
    # Record audio
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    # Convert to numpy array
    audio_npy = audio.flatten()

    # Save the audio to a temporary file 
    with open('temp.wav', 'w', encoding="utf-8") as file:
        file.write(np.array2string(audio))
    
    # Transcribe and translate the audio
    result = model.transcribe("temp.wav", task="translate", language="ro")
    
    print("Translation (English):", result['text'])
    

if __name__ == "__main__":
    while True:
        listen(duration=5)
        time.sleep(1)  # Pause before listening again
