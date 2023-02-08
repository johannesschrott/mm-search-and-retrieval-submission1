Select
    count(song2Id) as kOccurencesAt10, song2Id, featureType, similarityType
from (select * from (
    select song1Id,
           song2Id,
           similarityValue,
           similarityType,
           featureType,
           row_number() over (PARTITION BY song1Id order by similarityValue desc) as place
      from song_similarities
      where similarityType = 1
        and featureType = 0)
    where place <= 10)
group by song2Id, featureType, similarityType;