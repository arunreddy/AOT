
cd ../dataset/archive

tar xvf aclImdb_v1.tar.gz -C ../raw/
unzip amazon-fine-food-reviews.zip -d ../raw/amazon-fine-food-reviews
unzip amazonreviews.zip -d ../raw/amazonreviews
bunzip2 ../raw/amazonreviews/train.ft.txt.bz2
unzip cats_dogs.zip -d ../raw/cats_dogs
unzip convex.zip -d ../raw/convex
tar xvf GFK_v1.tar.gz -C ../raw/


extract_gz(){
  file_name=`echo $1 | cut -d'.' -f1`
  mkdir -p "../raw/$file_name"
  cp $1 "../raw/$file_name/"
  gunzip "../raw/$file_name/$1"
  echo "File $1 extracted to ../raw/$file_name/"
}

extract_gz Health.txt.gz
extract_gz "Home_&_Kitchen.txt.gz"
extract_gz "Tools_&_Home_Improvement.txt.gz"
extract_gz Video_Games.txt.gz
