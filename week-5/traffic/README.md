# Experimentation process

I have loaded the images as a dataset and started experimenting with sequential NN building in `get_model()`

1. Copied the model source code from lecture used for MNIST demonstration
   - Changed the input shape to match the traffic sign images (32, 32, 3)
   - Changed the output layer to have 43 neurons (one for each class) and soft
   - Of course, it was not accurate but gave me a starting point (5% accuracy)
2. Doubled the layer of conv-maxpool to experiment deep detalization
   - Damn, it increased to accuracy: 0.9697 - loss: 0.1091
3. Increased the density of layers after flattening from 128 to 512
   - Accuracy: 0.9335 - loss: 0.2528
   - It was a bit worse, so I tried 256 and got worse results, reverted to 128
4. Tried 20 Epochs and test 30%
   - accuracy: 0.9581 - loss: 0.1628
   - Reverted to 10 Epochs and 40% test
5. Did Conv32 -> MaxPool2_2 -> Conv64 -> MaxPool2_2 and Dense 256
   - Similar result to the max record

Stopping experimentation on 97% accuracy as it is already very good. I will be back with NN learned without bruteforce : )