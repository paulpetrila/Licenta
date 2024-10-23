import whisper

model = whisper.load_model('base')
out = model.transcribe('temp.wav', language='Romanian')
