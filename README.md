# Cosmos-AV-Sample Toolkits
This repo provides toolkits for generating HD map / bounding box / LiDAR condition video from the RDS-HQ dataset for [**Cosmos-Transfer1-7B-Sample-AV**](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV). It also provides conversion scripts from other datasets(e.g., Waymo Open Dataset) to RDS-HQ format.

## Install

```bash
conda env create -f environment.yaml
conda activate cosmos-av-toolkits
```

## Usage
You can use `render_from_rds_hq.py` to render the HD map + bounding box / LiDAR condition videos from RDS-HQ dataset
```bash
# single process
python render_from_rds_hq.py -i <RDS_HQ_FOLDER> -o <OUTPUT_FOLDER> [--skip hdmap] [--skip lidar]
```
You can add `--skip hdmap` or `--skip lidar` to skip the rendering of HD map and LiDAR, respectively.

You can also run this script in multiple processes to speed up the rendering process.
```bash
# multiple processes
torchrun --nproc_per_node=32 render_from_rds_hq.py -i <RDS_HQ_FOLDER> -o <OUTPUT_FOLDER> [--skip hdmap] [--skip lidar]
```
Set `nproc_per_node` to the number of processes you want to use.

**RDS-HQ Rendering Results**
![RDS-HQ Rendering Results](./assets/rds_hq_render.png)

## Convert from Other Dataset

### Waymo Open Dataset
Parsing tfrecords from Waymo Open Dataset requires extra dependencies; install it with
```bash
pip install waymo-open-dataset-tf-2-11-0==1.6.1
```

#### Step 1: Convert Waymo Open Dataset to RDS-HQ format
First, convert the Waymo Open Dataset to RDS-HQ format. Suppose you have a folder with Waymo Open Dataset's tfrecords, you can convert it to RDS-HQ format by:
```bash
python convert_waymo_to_rds_hq.py -i <WAYMO_TFRECORDS_FOLDER> -o waymo_demo -n 32
```
Here we specify the output folder as `waymo_demo`; you can change it to any other one. You can also increase the number of workers (`-n`) to your CPU cores to speed up the conversion.

#### Step 2: Render HD map + bounding box / LiDAR condition video from RDS-HQ format
Since we have converted Waymo Open Dataset's map labels and LiDAR points into the RDS-HQ format, we can render the HD map + bounding box / LiDAR condition video using the same script. The cameras in Waymo's open dataset are pinhole camera models. To align with our model's training domain, we suggest using the same f-theta camera intrinsics to do the projection. We provide a default f-theta camera intrinsics in `config/default_ftheta_intrinsic.tar`, and you can render with f-theta camera by:
```bash
python render_from_rds_hq.py -d waymo -i waymo_demo -o waymo_demo_render_ftheta -c ftheta
```
here `-d` is the dataset name, you can find its configuration in `config/dataset_waymo.json`, `-i` is the input folder, `-o` is the output folder. `-c` is the camera type, default it will be `ftheta` camera.

(Optional) We also provide the pinhole camera rendering, which will be useful if you want to finetune the model on your own dataset with pinhole camera.
```bash
python render_from_rds_hq.py -d waymo -i waymo_demo -o waymo_demo_render_pinhole -c pinhole
```

You can also run this script in multiple processes to speed up the rendering process.
```bash
torchrun --nproc_per_node=32 render_from_rds_hq.py -d waymo -i waymo_demo -o waymo_demo_render_ftheta 
```
Set `nproc_per_node` to the number of processes you want to use.

**Waymo Rendering Results (use ftheta intrinsics in RDS-HQ)**
![Waymo Rendering Results](./assets/waymo_render_ftheta.png)


**Waymo Rendering Results (use pinhole intrinsics in Waymo Open Dataset)**
![Waymo Rendering Results](./assets/waymo_render_pinhole.png)

