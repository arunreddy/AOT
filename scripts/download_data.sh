mkdir -p ./raw
cd ./raw

echo "Office-Caltech data set"
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=10780rJdxSGvSvJJuhVPd9WtHVeVOFf1u' -O GFK_v1.tar.gz
tar xvf GFK_v1.tar.gz
rm GFK_v1.tar.gz