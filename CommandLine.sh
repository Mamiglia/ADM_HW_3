countries=("Italy" "Spain" "France" "England" "United_States")
for country in ${countries[@]};
do
  country=${country/_/ } # to change 'United_States' into 'United States'

  echo Number of places that can be found in ${country} :
  nb_places=$(cut -f 3,4,8 places.tsv | grep "${country}" | wc -l)
  echo $nb_places

  i=0
  for numPeopleVisited in $(cut -f 3,4,8 places.tsv | grep "${country}" | cut -f 1); do
    i=$(( $i+$numPeopleVisited ))
  done
  echo Average visit of the places of ${country} :
  i=$(( $i/$nb_places ))
  echo $i
  
  k=0
  for numPeopleWant in $(cut -f 3,4,8 places.tsv | grep "${country}" | cut -f 2); do
    k=$(( $k+$numPeopleWant ))
  done
  echo People that want to visit the places of ${country} :
  echo $k

  echo
done
