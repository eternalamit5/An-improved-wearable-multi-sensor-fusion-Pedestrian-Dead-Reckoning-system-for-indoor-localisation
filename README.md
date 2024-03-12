# An improved wearable multi sensor fusion Pedestrian Dead Reckoning system for indoor localisation of Human
A solution for indoor positioning based on Pedestrian Dead Reckoning(PDR), with map matching using chest mounted IMU.

![pdr_video__27-09-2021_18-30-08__SparkVideo (1)](https://user-images.githubusercontent.com/44448083/134951489-0081450e-fe2b-4014-aaf6-d17f931e784e.gif)

The above Map is being created using the BIBA-Bremer Institut für Produktion und Logistik Bremen, Germany

# Concept Used:
- Step detection, step length estimation.
- Motion classification using machine learning method Multi-Layer Perceptrons (MLP).
- Design of Particle Filter with Map matching algorithm for pedestrian localisation and heading correction using Robust Adaptive Kalman Filter.


# System requirement
Hardware: IMU Xsens MTi-3 AHRS

Environment Required:

1. Python 3.6.7

2. scipy 1.0.0

3. numpy 1.14.0

4. OpenCV 3.4.2

5. python-osc 1.7.0

Recommended to create the environment easily using Anaconda

```
conda create -n py36 python=3.6 scipy=1.0.0 numpy=1.14.0 opencv
pip install python-osc
```


