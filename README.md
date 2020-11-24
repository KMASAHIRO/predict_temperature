# predict_temperature
This repo includes all the files about training a Keras CNN model that predicts temperature and running it on raspberry pi 4 with Edge TPU (Coral).

# About
I made and trained a keras 1D-CNN model that predicts the next temperature based on 30 temperature data.
The dataset I used is one day's average temperature for about 20 years provided by [Japan Meteorological Agency](https://www.data.jma.go.jp/obd/stats/etrn/index.php).
After that, I converted the model to TensorFlow Lite models that run on Raspberry pi 4, then compiled one of the model 
so that it should run with Edge TPU (Coral).
Using this model and test data, I predicted the next day's temperature.
In addition, I predicted Raspberry pi's CPU temperature.

This repo includes all the files I used to take these steps.

# Envs
### Training, converting and compiling the model
- Google Colab
- Tensorflow 2.3.1
- edgetpu_compiler 15.0.340273435

### Running the model
- Raspbian 10 (Raspberry pi 4)
- tflite_runtime 2.1.0.post1
- edgetpu_runtime 2.14.1

# Explanation of each directory

## cnn_train
This directory includes an ipynb file I ran on Google Colab, three TFLite models, and the training dataset of csv.

In ```cnn_weather.ipynb```, I made and trained the keras model. Then, I converted the model to two TFLite models,
one of which is quantized and the other is not. The quantized model was compiled to run on Edge TPU.

```cnn_weather_lite.tflite``` is a non-quantized TFLite model file made in ```cnn_weather.ipynb```.

```cnn_weather_lite_quantized.tflite``` is a quantized TFLite model file made in ```cnn_weather.ipynb```.

```cnn_weather_lite_quantized_edgetpu.tflite``` is a compiled TFLite model file made in ```cnn_weather.ipynb```.

``temp.csv``` is the training dataset that contains one day's average temperature for about 20 years.

## predict_cputemp
This directory includes three python files I ran on Raspberry pi 4 to predict CPU temperature.

```predict_cpu.py``` gets CPU temperature of a Linux machine every second for 30 seconds and predicts the next temperature
on the non-quantized model for CPU.
This code outputs the input temperatures, the predicted next temperature, the real next temperature, and time to run the code.

```predict_cpu_quantized.py``` gets CPU temperature of a Linux machine every second for 30 seconds and predicts the next temperature
on the quantized model for CPU.
This code outputs the input temperatures, the predicted next temperature, the real next temperature, and time to run the code.


```predict_edgetpu.py``` gets CPU temperature of a Linux machine every second for 30 seconds and predicts the next temperature
on the compiled model for Edge TPU.
This code outputs the input temperatures, the predicted next temperature, the real next temperature, and time to run the code.

## predict_test
This directory includes three python files I ran on Raspberry pi 4 to predict the next day's temperature
using test data, and the test data of csv.

```cpu_quantized_time.py``` predicts the next day's temperature on the quantized model for CPU and outputs the temperature
and time to predict.

```cpu_time.py``` predicts the next day's temperature on the non-quantized model for CPU and outputs the temperature
and time to predict.

```edgetpu_time.py``` predicts the next day's temperature on the compiled model for Edge TPU and outputs the temperature
and time to predict.

```test.csv``` is the test data that contain one day's average temperature for recent 30 days.
