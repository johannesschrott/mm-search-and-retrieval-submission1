select * from (
    select song1Id,
           song2Id,
           similarityValue,
           similarityType,
           featureType,
           row_number() over (order by similarityValue desc) as place
      from song_similarities
      where similarityType = 1
        and featureType = 0)
    where place <= 10;