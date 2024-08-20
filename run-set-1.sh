python compress_net.py \
        -n networks/mnist/mnist_relu_3_50.onnx \
        -a 20 \
        -s 2000 \
        -b 5000 \
        -l 0.05 \
        --force-no-end-relu True \
        -c 0
