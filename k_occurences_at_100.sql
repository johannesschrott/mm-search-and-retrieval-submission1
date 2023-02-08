Select
    count(song2Id)  as kOccurencesAt100, song2Id, featureType, similarityType
from song_similarities
where featureType = 0
  and similarityType = 1
group by song2Id, featureType, similarityType;