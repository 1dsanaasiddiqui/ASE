This repository contains the code, benchmarks and experimental data for the paper "Learning DNN Abstractions using Gradient Descent".

#Installation Instructions:
Clone the repository and update all submodules. Then, create a new conda
environment via:
conda env remove --name learning-merges
conda create --name learning-merges
conda activate learning-merges
pip install -r requirements.txt

#Running All Experiments:
Use the script mnist_small_ep_batch_3_50.sh to run the experiments using our method.

Once the run is complete, table.txt will contain a summary of the results.




