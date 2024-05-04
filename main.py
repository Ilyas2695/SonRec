import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Инициализация подключения к Spotify
client_id = ''
client_secret = ''
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Получение данных о треке
track_id = '4k6Uh1HXdhtusDW5y8Gbvy'
track_info = sp.track(track_id)
print(track_info['name'], "by", track_info['artists'][0]['name'])

# Скачивание превью аудиофайла трека
preview_url = track_info['preview_url']
if preview_url:
    urllib.request.urlretrieve(preview_url, 'track.mp3')
    # Загрузка аудиофайла
    try:
        y, sr = librosa.load('track.mp3', sr=None)
        # Получение спектограммы
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # Отображение спектограммы
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error loading audio file: {e}")
else:
    print("Preview not available for this track.")
