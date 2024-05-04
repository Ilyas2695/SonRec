import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Инициализация подключения к Spotify
client_id = 'd67e87eab19146338c701fc80d622899'  # Замени на свой Client ID
client_secret = '13dab873a5384a39a35ab8d22e6c77dd'  # Замени на свой Client Secret
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Получение информации о треке по его ID
track_id = '3n3Ppam7vgaVa1iaRUc9Lp'  # Пример ID трека, можно заменить на другой
track_info = sp.track(track_id)
print(track_info['name'], "by", track_info['artists'][0]['name'])

# Скачивание превью аудиофайла трека
preview_url = track_info['preview_url']
if preview_url:
    urllib.request.urlretrieve(preview_url, 'track.mp3')
else:
    print("Preview not available for this track.")

# Загрузка аудиофайла
y, sr = librosa.load('track.mp3', sr=None)

# Получение спектограммы
spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# Отображение спектограммы
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
