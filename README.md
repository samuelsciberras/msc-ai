<p align="center"><a align="center" href="https://www.um.edu.mt/"><img src="UoM.png" alt="Logo" width="100" height="100"></a></p>

<h2 align="center">Land Vehicle Speed Estimation in an Augmented Reality Space using Computer Vision</h2>
<h3 align="center">Samuel Sciberras</h3>
<h4 align="center">Supervised by Dr Vanessa Camilleri<br>Co-supervised by Mr Dylan Seychell</h3>
<h5 align="center">Department of Artificial Intelligence<br>Faculty of ICT<br>University of Malta</h5>

<br><br>
<p align="center"><i>A dissertation submitted in partial fulfilment of the requirements for the degree of M.Sc. in Artificial Intelligence.</i></p>
<p align="center">May 2021</p>
<hr>


<br><br>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#how-to-use">How To Use</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br><br>


<!-- BUILT WITH -->
## Built With

* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)


<!-- Prerequisites -->
## Prerequisites

1. [Download](https://www.anaconda.com/) and install Anaconda distribution for Python 3.8
2. (Optional*) [Download](https://developer.nvidia.com/cuda-10.2-download-archive) and install CUDA Toolkit 10.2
3. (Optional*) [Download](https://developer.nvidia.com/rdp/cudnn-archive) and install cuDNN compatible with version CUDA 10.2

<i>* Required for real-time processing<br>* Requires Nvidia CUDA compatible GPU</i>



<!-- Installation -->
## Installation

1. Clone repository
   ```sh
   git clone https://github.com/samuelsciberras/msc-ai
   ```
2. Create conda environment
   ```sh
   conda env create --file yv4-dsort-speedest.yml
   ```
3. Download YOLOv4 trained models [here](https://drive.google.com/file/d/1YGPwcSlXKjyvjSrA2D6m2bvTClRxm73a/view?usp=sharing) and place them in directory /System/checkpoints/
   
<!-- Running -->
## How To Use

1. Activate conda environment
   ```sh
   conda activate yv4-dsort-speedest
   ```
2. Run 
   ```sh
   cd System/
   python main.py
   ```


<!-- Contact-->
## Contact

Samuel Sciberras - samuelsciberras96@gmail.com
