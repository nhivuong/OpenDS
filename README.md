# OpenDS
Open-source implementation of Discriminant Center-surround Hypothesis on Saliency Detection, based on the paper “On the plausibility of the discriminant center-surround hypothesis for visual saliency” by Dashan Gao; Vijay Mahadevan and Nuno Vasconcelos in 2008 https://ieeexplore.ieee.org/abstract/document/4408851 

This is a project work dedicated to EECS 4422 - Computer Vision course at York University, Fall 2019. 
Within this repo, there are the following files:
````
|
|--report
|      |--final_report.pdf
|      |--miscellaneous files of .tex and .png
|--testimages
|      |--test images in png format
|--OpenDS.ipynb
|--main.py
|--util.py
|--GaborKern.py
|--featureMap.py
|--README.md
````

### Installation Instruction: 
Please pull/ clone this repo

### Requirements: 
Make sure python 3.6 and the following packages are installed:
```` 
cv2
numpy
scipy
matplotlib
````
Version differences don't cause major issue.

### How to run Python files: 
From terminal, run the command: ````python3 main.py````

This python code will write the result saliency map into your current dir, check for file: ```sal_map.jpg``` after executing.

If you wish to change the input image, please update this code line in ```main.py```

```img = cv2.imread("./testimages/test3.png", 1)```

### How to run JupyterNotebook file: 
From your Jupyter home, open ````OpenDS.ipynb````. This notebook should already contain some output to display through matplotlib (jupyter code doesn't write output map into current dir).


### Credit: 
Itti-koch implementation comes from https://github.com/shreelock/gbvs/blob/master/ittikochneibur.py

