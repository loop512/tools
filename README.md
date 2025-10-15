# tools
Tools with analysis, visualization, and other functionalities for VoIP steganalysis  

The '📁 BayesianNetworks/' directory contains code for building Bayesian networks:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 Read_data.py': Reads data from speech samples and saves the extracted codewords into CSV files.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 main.py': Loads codewords from the CSV files, constructs the Bayesian networks, and saves the resulting network structures as images.  

The '📁 Barchart/' directory contains code for visualizing (Bar chart) codeword features.:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 729CNV_English_0.5_0.3s.csv': Stores labels for all data.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 data.npy': Stores codeword features extracted from all data. This file exceeds GitHub's 25 MB limit. Kindly contact zhangcheng@cumt.edu.cn for the original file if you are interested.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 main.py': Loads codeword features and their corresponding labels, and generates a 3D visualization of the features.  
  
The '📁 Heatmap/' directory contains code for visualizing (Heatmap) codeword features.:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📁 data/': Stores the original data.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📁 code/': Contains code for visualizing (Heatmap) codeword features.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - '📄 main.py': Read data from the 'data/' folder and generate heatmaps based on the mutual information (MI) and distance correlation (dCor) matrices.


**More tools will be further updated in the near future.**

We release these tools, which have been used in the following papers, to enable further research. Proper citation of the associated publications would be greatly appreciated when these tools are used：
1. C. Zhang, S. Jiang, Z. Chen, and J. Qian, "MSSN: Multi-Stream Steganalysis Network for Detection of QIM-based Steganography in VoIP Streams," in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2025.3612372.
2. To be updated
3. To be updated
