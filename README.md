# Captioning satellitar images with CLIP + GPT2

## Downloading datasets 

For all the datasets the captions are already present in this repo. The images should be downloaded.

#### RSICD Captions

Can be found here https://github.com/201528014227051/RSICD_optimal. The folder structure is the following:
```
| data
|  ├── rsicd
│  │  ├── RSICD_images
│  │  │  ├── 00001.jpg
│  │  │  ├── 00002.jpg
│  │  │  ├── ...
│  │  ├── dataset_rsicd.json
```

#### UCM Captions

Can be found here https://github.com/201528014227051/RSICD_optimal. The folder structure is the following:
```
| data
|  ├── ucm
│  │  ├── images
│  │  │  ├── 1.tif
│  │  │  ├── 2.tif
│  │  │  ├── ...
│  │  ├── dataset.json
```

#### Sydney Captions

Can be found here https://github.com/201528014227051/RSICD_optimal. The folder structure is the following:
```
| data
|  ├── sydney
│  │  ├── images
│  │  │  ├── 1.tif
│  │  │  ├── 2.tif
│  │  │  ├── ...
│  │  ├── filenames
│  │  │  ├── descriptions_SYDNEY.txt
│  │  │  ├── filenames_test.txt
│  │  │  ├── filenames_train.txt
│  │  │  ├── filenames_val.txt
```

#### NWPU Captions

Can be found here: https://github.com/HaiyanHuang98/NWPU-Captions. The folder structure is the following:
```
| data
|  ├── nwpu
│  │  ├── images
│  │  │  ├── airplane
│  │  │  │  ├── airplane_001.jpg
│  │  │  │  ├── ...
│  │  │  ├── bridge
│  │  │  │  ├── bridge_001.jpg
│  │  │  │  ├── ...
│  │  │  ├── ...
│  │  ├── dataset_nwpu.json
```

## Training

To train the model, you have first to train clip alone. The trainind is done using all the datasets combined.

```bash
python train_clip.py --lr 1e-6 --batch_size 32 --epochs 10
```

Then you can train the GPT2 model:

```bash
python train_decoder.py --lr_gen 1e-6 --lr_adapter 1e-4 --batch_size 32 --epochs 10
```

## Evaluation

To evaluate the model. It will evaluate the model on each dataset and printing the metrics.

```bash 
python evaluate.py
```

## Inference

To generate captions for a specific image, you can use the following command:

```bash
python caption.py --image_path path/to/image.jpg
```

