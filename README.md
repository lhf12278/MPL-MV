## Mutual prediction learning and mixed viewpoints for unsupervised-domain adaptation person re-identification on blockchain

  
## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
* PrettyTable (```pip install prettytable```)
* GPU Memory >= 10G
* Memory >= 10G

## Dataset Preparation
* Market-1501 ([Project](http://www.liangzheng.com.cn/Project/project_reid.html), [Google Drive](https://drive.google.com/open?id=1M8m1SYjx15Yi12-XJ-TV6nVJ_ID1dNN5))
* DukeMTMC-reID ([Project](https://github.com/sxzrt/DukeMTMC-reID_evaluation), [Google Drive](https://drive.google.com/open?id=11FxmKe6SZ55DSeKigEtkb-xQwuq6hOkE))
* MSMT17 ([Project](https://www.pkuvmc.com/dataset.html), [Paper](https://arxiv.org/pdf/1711.08565.pdf), Google Drive \<please e-mail me for the link\>)


## Run
#### Train on Market-1501/DukeMTMC-reID/MTMC17
```
python main.py --mode train \
    --train_dataset market --test_dataset duke \
    --market_path /path/to/market/dataset/ \
    --duke_path /path/to/duke/dataset/ \
    --output_path ./results/market/ 
python main.py --mode train \
    --train_dataset duke --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --duke_path /path/to/duke/dataset/ \
    --output_path ./results/duke/
python main.py --mode train \
    --train_dataset duke --test_dataset msmt17 \
    --duke_path /path/to/duke/dataset/ \
    --msmt17_path /path/to/msmt17/dataset/ \
    --output_path ./results/duke/
python main.py --mode train \
    --train_dataset market --test_dataset msmt17 \
    --market_path /path/to/market/dataset/ \
    --msmt17_path /path/to/msmt17/dataset/ \
    --output_path ./results/duke/

```
