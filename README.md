Lane_RSR
========
Overview
--------
![1](https://user-images.githubusercontent.com/62361339/147277573-87d250f2-fc52-4b87-9bca-b8dd4d1b9637.gif)
![2](https://user-images.githubusercontent.com/62361339/147277615-b4d9691a-d588-4bed-83e9-b1ca13fb3287.gif)
![3](https://user-images.githubusercontent.com/62361339/147277622-71644a2d-9d05-4abc-a542-39ac957b74fe.gif)

Lane_RSR is a cascaded framework for robust lane detection.

It consists of reconstruction module, segmentation module, and restoration module.

This implementation uses ENet-SAD model as segmentation module.

![framework](https://user-images.githubusercontent.com/62361339/147280796-7fc24e41-f7a9-4dbf-9047-0272e755a775.PNG)


#### Dependencies
```bash
pip3 install -r requirements.txt
```

#### Datasets
Lane_RSR is trained and tested on CULane dataset.

CULane dataset is available in <https://xingangpan.github.io/projects/CULane.html>

Lane Reconstruction
-------------------
```bash
cd UDLR
```

train
```bash
python3 main.py --mode = train
```

test
```bash
python3 main.py --mode = test
```


Lane Restoration
-----------------
TBU
