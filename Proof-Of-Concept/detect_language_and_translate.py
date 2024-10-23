import whisper
from pydub import AudioSegment

# Record an audio sample to pass to whisper
sample_rate = 44100
frequency = 440
length = 5


model = whisper.load_model("base")



audio = whisper.load_audio("./bunasiua.mp3")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)

_, probs = whisper.detect_language(model, mel)

print(probs)
print(f"Detected language:{max(probs, key= lambda item:item[1])}")

# options = whisper.DecodingOptions()

