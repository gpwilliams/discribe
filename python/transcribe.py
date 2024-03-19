# load packages ---------------------------------------------------------------------

import os
# https://github.com/linto-ai/whisper-timestamped
import whisper_timestamped as whisper
import pandas as pd

# load model ---------------------------------------------------------------------

model = whisper.load_model("large-v3", device="cpu")

# user inputs ---------------------------------------------------------------------

user_folder = "vera" # specify the folder path
model_language = "en"

# sort paths ---------------------------------------------------------------------

# join user folder with the input
input_path = os.path.join("input", user_folder)

# get the output paths
transcription_output_path = os.path.join("output/transcriptions", user_folder)
words_output_path = os.path.join("output/words", user_folder)

# create the output folders if missing
if not os.path.exists(transcription_output_path):
    os.makedirs(transcription_output_path)
                
if not os.path.exists(words_output_path):
    os.makedirs(words_output_path)

# get a list of all .wav files in the folder
wav_files = [file for file in os.listdir(input_path) if file.endswith(".wav")]

# transcribe ---------------------------------------------------------------------

total_files = len(wav_files)

for file in wav_files:
    # print progress to console
    print(f"Transcribing file {file} of {total_files}...")
    
    # construct the full file path
    file_path = os.path.join(input_path, file)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    result = whisper.transcribe(model, audio, language=model_language, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    
    # append the result to the transcriptions DataFrame
    file_name = os.path.splitext(file)[0]

    for segment in result['segments']:
        df_transcriptions = pd.DataFrame({
            'File': [file_name], 
            'Text': [result['text']], 
            'Avg_Logprob': [segment['avg_logprob']],
            'Confidence': [segment['confidence']]
            })
        df_transcriptions.to_csv(os.path.join(transcription_output_path, file_name + '.csv'), index=False)

    for segment in result['segments']:
        # add in the word number
        df_words = pd.DataFrame(segment['words'])
        df_words['Word_Number'] = df_words.index + 1  # add the word number
        df_words['File'] = file_name  # add the file name to each row
        df_words.to_csv(os.path.join(words_output_path, file_name + '.csv'), index=False)
