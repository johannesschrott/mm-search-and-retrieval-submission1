from evaluation.Genres import Genres
import data_statistics as stats

genres = Genres()
top5, avgGenresTrack, avgTracksSharing1Genre, meanTracksPerGenre, medianTracksPerGenre = stats.getStats(genres.genres, 5)
print('top 5: ', top5)
print('Average number of genres per track: ', avgGenresTrack)
print('Average number of tracks that share at least one genre: ', avgTracksSharing1Genre)
print('Average number of tracks per genre (mean): ', meanTracksPerGenre)
print('Average number of tracks per genre (median): ', medianTracksPerGenre)