This is a personal implementation of some ego-motion estimation models using millimeter wave radar.

## Packages below are required
```
numpy
nibabel
torch
scipy
filterpy
tqdm
numba
scikit-learn
multiprocessing
```

## Radar-inertial

This is a personal implementation of Radar-Inertial Ego-Velocity Estimation for Visually Degraded
Environments which is published on ICRA 2020.
```
@inproceedings{kramer2020radar,
  title={Radar-inertial ego-velocity estimation for visually degraded environments},
  author={Kramer, Andrew and Stahoviak, Carl and Santamaria-Navarro, Angel and Agha-Mohammadi, Ali-Akbar and Heckman, Christoffer},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5739--5746},
  year={2020},
  organization={IEEE}
}
```

If you would like to implement your own data, modify file ```./utility/data_process.py```, modify the output path in this file as well, then you can run
```
python run.py
```
Note that the program runs in parallel in default, the number of cores is set by
```
pool = mp.Pool()
```
remove the comment to run the program in serial.

## Milli-RIO
This is a personal implementation of Milli-RIO: Ego-motion estimation with low-cost millimetre-wave radar.

```
@article{almalioglu2020milli,
  title={Milli-RIO: Ego-motion estimation with low-cost millimetre-wave radar},
  author={Almalioglu, Yasin and Turan, Mehmet and Lu, Chris Xiaoxuan and Trigoni, Niki and Markham, Andrew},
  journal={IEEE Sensors Journal},
  volume={21},
  number={3},
  pages={3314--3323},
  year={2020},
  publisher={IEEE}
}
```
The path of data is a bit complicated, carefully modify the files in ```/utility```. Then you can train Bi-LSTM by running
```
python train_rnn.py
```
The model will be saved to ```./models```, when testing, whoose the relating model.
```
python test.py
```

## MilliEgo
This is a personal code of data transformation for [milliEgo](https://github.com/ChristopherLu/milliEgo). It's used to transform normal
millimeter wave radar point cloud into depth image according to the paper.

Note that the program runs in parallel in default, the number of cores is set by
```
pool = mp.Pool()
```
Also, it used ```numba``` to accelerate loops, feel safe to ignore the warnings.
