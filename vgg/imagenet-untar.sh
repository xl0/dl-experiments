

IMAGENET_DIR=/root/datasets/ImageNet

cd $IMAGENET_DIR/train
find . -name "*.tar" | while read NAME
do
    echo "${NAME} -> ${NAME%.tar}"
    mkdir -p "${NAME%.tar}" || exit 1
    tar -xf "${NAME}" -C "${NAME%.tar}" || exit 1
    rm "${NAME}"
done


cd $IMAGENET_DIR/val
echo "Extracting ILSVRC2012_img_val.tar"
tar -xf ILSVRC2012_img_val.tar || exit 1
echo "Renaming files"
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash || exit 1
rm ILSVRC2012_img_val