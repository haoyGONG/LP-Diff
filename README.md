# LPDGAN


### LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate [[Paper Link]]([https://arxiv.org/abs/2404.13677](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html))
[Haoyan Gong], [Zhenrong Zhang], [Yuzheng Feng], [Anh Nguyen] and [Hongbin Liu*]


## Real-World Degraded License Plate Results

**Comparison with the other methods.**

<img src="https://github.com/haoyGONG/LP-Diff/tree/main/figs/results_with_word.jpg" width="800"/>



### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
```

## Download Dataset
- The MDLP dataset is available at [Google Drive](https://drive.google.com/file/d/1UpECGcWcF92z-P6pJ9couzGTXb1TMHqk/view?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1Aphb_jIx_0tRR71BBbwVwA?pwd=1ebm) (access code: 1ebm).  


## How To Train
- Refer to `./run.py` for model training and testing.
- The training command is like
```
python run.py -p train -c .\config\LP-Diff.json
```
- The validation command is like
```
python run.py -p val -c .\config\LP-Diff.json
```

The training logs and weights will be saved in the `./experiments` folder.

## Acknowledgement
This project is built mainly based on the excellent [ResDiff](https://github.com/LYL1015/ResDiff/tree/master) codeframe. We appreciate it a lot for its developers.

## Contact
If you have any question, please email m.g.haoyan@gmail.com.
