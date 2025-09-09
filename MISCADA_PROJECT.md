# UnSAM: Unsupervised Segment Anything Model API

## Installation
See [installation instructions](INSTALL.md).

**Install dependencies**

```bash
pip install -r requirements.txt
pip install -e segment_anything
pip install -e detectron2
```

## The weights used

```bash
checkpoints/unsam_sa1b_4perc_ckpt_200k.pth
checkpoints/unsam_plus_promptable_sa1b_1perc_ckpt_100k.pth
```
Download link：
https://drive.google.com/file/d/12DvjnXIQsOtBSAAEicd9uhW0TCpnMFyZ/view
https://drive.google.com/file/d/1M3lOnSOutQRK4IqBkc3e4vGZ-u2oTkeW/view

## Start Service 
```py
uvicorn unsam_service:app --host 0.0.0.0 --port 8008 --reload

```

## Test interface
```py
python test_point.py
```


