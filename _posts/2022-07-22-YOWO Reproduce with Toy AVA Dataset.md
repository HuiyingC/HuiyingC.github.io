---
title: "YOWO Reproduce with Toy AVA Dataset"
date: 2022-07-22T0:03:30-07:00
categories:
  - blog
tags:
  - paper
  - implementation
  - AVA
---



# Data Preparation

## Video and annotation preparation

[AVA offical project page](https://research.google.com/ava/)
[Guide in SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md)
[Guide in Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks/blob/main/DATASET.md)

We assume that the AVA dataset is placed at data/ava with the following structure.

ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv

Video wget links:
```
https://s3.amazonaws.com/ava-dataset/trainval/[file_name]
https://s3.amazonaws.com/ava-dataset/test/[file_name]
```

The lists of file names [train/val](https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt), [test](https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_test_v2.1.txt)
Annotations: [ava_v2.2.zip](https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip)

Prepare above structure by following scripts.

1. Download videos (These video files take 157 GB of space).
./download_videos.sh
```bash
DATA_DIR="../../data/ava/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
```

2. Cut each video from its 15th to 30th minute.
./cut_videos.sh
```bash
IN_DATA_DIR="../../data/ava/videos"
OUT_DATA_DIR="../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
```

3. Extract frames (These frames take 392 GB of space).
./extract_frames.sh  (ffmpeg installed)
```bash
IN_DATA_DIR="../../data/ava/videos_15min"
OUT_DATA_DIR="../../data/ava/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
```

4. Download annotations. [[full .tar file](https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar)]
./download_annotations.sh
```bash
DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
```

5. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv)) and put them in the `frame_lists` folder (see structure above).

6. Download person boxes ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv), [test](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv)) and put them in the `annotations` folder (see structure above). For personal detector go [here](https://github.com/facebookresearch/video-long-term-feature-banks/blob/main/GETTING_STARTED.md#ava-person-detector).

## Download backbone pretrained weights

Darknet-19 weights can be downloaded via:
`wget http://pjreddie.com/media/files/yolo.weights`

ResNeXt ve ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M).

## Pretrained YOWO models

Pretrained models for AVA dataset can be downloaded from [here](https://drive.google.com/drive/folders/1g-jTfxCV9_uNFr61pjo4VxNfgDlbWLlb).


# Running the code

- Modify [ava.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ava.yaml)
- AVA training:
`python main.py --cfg cfg/ava.yaml`

# Validating the model

For AVA dataset, after each epoch, validation is performed and frame-mAP score is provided.
`python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER`

# Running on a test video with pretrained model

AVA:
`python test_video_ava.py --cfg cfg/ava.yaml`


<hr style="border:2px solid gray"></hr>
# Work with two toy datasets from AVA

Refer to blog working on SlowFast model with AVA dataset: [link](https://blog.csdn.net/WhiffeYF/article/details/115444694?spm=1001.2014.3001.5501)

Structure annotation files as below.
```
annotations
—person_box_67091280_iou90
------ava_detection_train_boxes_and_labels_include_negative_v2.2.csv
------ava_detection_val_boxes_and_labels.csv
—ava_action_list_v2.2_for_activitynet_2019.pbtxt
—ava_detection_val_boxes_and_labels.csv
—ava_train_v2.1.csv
—ava_train_v2.2.csv
—ava_val_excluded_timestamps_v2.2.csv
—ava_val_v2.2.csv
```

## Preprocessing:

1. `/annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv`

​		Only keep rows with `original_vido_id = 'N5UD8FGzDek' `

2. `/annotations/person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv` 

​		Only keep rows with `vido_id = '1j20qq1JyX4'`

3. `/annotations/ava_train_v2.2csv`

​		Only keep rows with `original_vido_id = 'N5UD8FGzDek' `

4. `/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt`

​		Unchanged

5. `/annotations/ava_detection_val_boxes_and_labels.csv` (MISSING!)

​		Only keep rows with `vido_id = '1j20qq1JyX4'`

6. `/annotations/ava_val_excluded_timestamps_v2.2.csv`

​		Empty

7. `/annotations/ava_val_v2.2.csv`

​		Only keep rows with `vido_id = '1j20qq1JyX4'`









