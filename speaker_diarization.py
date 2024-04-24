import sys
sys.path.append(".")
sys.path.append("..")
import torch
import torchaudio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import contextlib
import wave
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import datetime
import subprocess
import wespeaker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from pydub import AudioSegment

def audio_crop(start, end, old_file, new_file, fmt='wav'):
    try:
        wav_file = AudioSegment.from_file(old_file)
        crop_file = wav_file[start:end]
        crop_file.export(new_file, format=fmt)
    except Exception as e:
        print("can't crop audio file: ' + old_file")

def determine_best_number_of_clusters(embeddings):

    # Define a range of possible clusters to test
    range_n_clusters = list(range(2, len(embeddings)))

    # Initialize the best score to a very low value
    best_score = -1

    best_n_clusters = 1
    
    # Iterate over the possible number of clusters
    for n_clusters in range_n_clusters:
        # Initialize the KMeans with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        
        # Fit the KMeans model and predict the cluster labels
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")
        # If the silhouette score is better than the current best score, update the best score and the best number of clusters
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
    
    return best_n_clusters, best_score

class SpeakerDiarizer:
    def __init__(self, num_speakers=2):
        self.num_speakers = num_speakers

        # NOTE: You would have to ensure that the necessary libraries are installed
        # and the models are available on the path where this class is used
        import whisper
        self.model = whisper.load_model('large')
        # self.embedding_model = PretrainedSpeakerEmbedding(
        #     "speechbrain/spkrec-ecapa-voxceleb",
        #     device=torch.device("cuda"))
        self.wespeaker_model = wespeaker.load_model("chinese")


    def segment_embedding(self, segment, path, duration):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        audio = Audio()
        waveform, sample_rate = audio.crop(path, clip)
        audio_crop(start * 1000, end *1000, path, 'audio_crop.wav')
        tensor = self.wespeaker_model.extract_embedding('audio_crop.wav')
        embedding = tensor.detach().cpu().numpy()
        return embedding, tensor
        # return self.embedding_model(waveform[None])


    def diarize(self, path):
        if path[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
            path = 'audio.wav'

        result = self.model.transcribe(path)
        segments = result["segments"]

        with contextlib.closing(wave.open(path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        embeddings = np.zeros(shape=(len(segments), 256))
        tensors = []
        for i, segment in enumerate(segments):
            embeddings[i], tensor = self.segment_embedding(segment, path, duration)
            tensors.append(tensor)

        embeddings = np.nan_to_num(embeddings)
        
        for i in range(len(segments)):
            for j in range(len(segments)):
                if i < j:
                   score = self.wespeaker_model.cosine_similarity(tensors[i], tensors[j])
                   print(f"Segment {i} and Segment {j} similarity: {score}")

        best_n_clusters, best_score = determine_best_number_of_clusters(embeddings)
        print("Best number of clusters:", best_n_clusters)
        print("Best silhouette score:", best_score)

        clustering = AgglomerativeClustering(self.num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        def time(secs):
            return datetime.timedelta(seconds=round(secs))

        transcript = ""
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                transcript += "\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n'
            transcript += segment["text"][1:] + ' '

        return transcript
