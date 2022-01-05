import os
import pickle
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from scipy.io import wavfile
from pydub import AudioSegment
from operator import itemgetter
import matplotlib.pyplot as plt
from collections import defaultdict

def convert_to_mono(file):
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export(file, format="wav")

def read_audio(file, verbose=False):
    sample_rate, data = wavfile.read(file)
    data = np.nan_to_num(data)

    if verbose:
        print("Sample Rate:", sample_rate)
        print("Data Shape:", data.shape)
        print("Song Length:", data.shape[0] / sample_rate, "seconds")
        
        intro = 10
        time = np.linspace(0, intro, intro * sample_rate)
        plt.plot(time, data[0:intro * sample_rate, 0], label="Data")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
    
    return sample_rate, data

def convert_domain(data, chunk_size):
    num_chunks = data.shape[0] // chunk_size
    freq_domain = []
    for i in range(num_chunks):
        chunk = []       
        for j in range(chunk_size):
            chunk.append(data[i * chunk_size + j])

        freq_domain.append(fft(chunk))
        
    return freq_domain

def get_index(freq, intervals):
    index = 0
    while intervals[index] <= freq:
        index += 1
    return index - 1
    
def signature_hash(f, fuz_factor):
    return (f[3] - (f[3] % fuz_factor)) * 100000000 + (f[2] - (f[2] % fuz_factor)) * 100000 + (f[1] - (f[1] % fuz_factor)) * 100 + (f[0] - (f[0] % fuz_factor))

def fingerprint(database, intervals, freq_domain, file):
    for i in range(len(freq_domain)):
        max_amplitudes = [0] * (len(intervals) - 1)
        frequencies = [0] * (len(intervals) - 1)
        for freq in range(intervals[0], intervals[-1]):
            index = get_index(freq, intervals)
            magnitude = abs(freq_domain[i][freq])
            if magnitude > max_amplitudes[index]:
                max_amplitudes[index] = magnitude
                frequencies[index] = freq

        signature = signature_hash(frequencies, 2)
        value = file + "," + str(i)
        database[signature].append(value)
    
def add_songs(database, chunk_size, intervals):
    for file in os.listdir("songs/"):
        if file.endswith(".wav") and not "test" in file:
            sample_rate, data = read_audio("songs/" + file)
            freq_domain = convert_domain(data, chunk_size)
            fingerprint(database, intervals, freq_domain, file)   
            
def match_hashes(database, intervals, freq_domain):
    rel_times = defaultdict(list)
    for i in range(len(freq_domain)):
        max_amplitudes = [0] * (len(intervals) - 1)
        frequencies = [0] * (len(intervals) - 1)
        for freq in range(intervals[0], intervals[-1]):
            index = get_index(freq, intervals)
            magnitude = freq_domain[i][freq]
            if magnitude > max_amplitudes[index]:
                max_amplitudes[index] = magnitude
                frequencies[index] = freq

        signature = signature_hash(frequencies, 2)
        if signature in database:
            for value in database[signature]:
                key = value.split(",")[0]
                song_time = int(value.split(",")[1])
                rel_times[key].append(abs(i - song_time))
    
    scores = []
    for key, value in rel_times.items():
        value = sorted(value)
        matches = len(value)
        if matches != 0:
            current = value[0]
            streak = 1
            for i in range(1, len(value)):
                if value[i] == current:
                    streak += 1
                if value[i] != current or i == len(value) - 1:
                    matches += streak - 1
                    streak = 1
                    current = value[i]
                
        scores.append([key, matches])
        
    scores = sorted(scores, key=itemgetter(1), reverse=True)
    return scores

def create_database(dir, chunk_size, intervals):
    for file in os.listdir(dir):
        if file.endswith(".wav"):
            convert_to_mono(dir + file)     

    database = defaultdict(list)
    add_songs(database, chunk_size, intervals)
    with open('database.pkl', 'wb') as f:
        pickle.dump(database, f)
    
    return database

def load_database():
    with open('database.pkl', 'rb') as f:
        database = pickle.load(f)
    return database
        
def match_song(song, database, chunk_size, intervals):
    convert_to_mono(song)
    sample_rate, data = read_audio(song)
    freq_domain = convert_domain(data, chunk_size)
    return match_hashes(database, intervals, freq_domain)    

def record_song(output_file, seconds):
    sample_rate = 44100
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()
    wavfile.write(output_file, sample_rate, recording)

chunk_size = 4096
intervals = [40, 80, 120, 180, 300]
dir = "songs/"
if os.path.isfile("database.pkl"):
    database = load_database()
    
    output_file = "recording.wav"
    seconds = 20
    record_song(output_file, seconds)
    matches = match_song(output_file, database, chunk_size, intervals)
    print("Match:", matches)
else:
    database = create_database(dir, chunk_size, intervals)