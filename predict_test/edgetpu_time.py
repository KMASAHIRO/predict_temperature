'''
Predict the next day's temperature based on the test
data(one day's average temperature for 30 days) using
the quantized model for edgetpu, and measure time
it took to predict.

Output is the predicted temperature and the measured
time.

Environments
Raspbian 10 (Raspberry pi4)
tflite_runtime 2.1.0.post1
edgetpu_runtime 2.1.4
'''

import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import time

def main(args):
    #interpret the quanitzed model for edgetpu
    interpreter = tflite.Interpreter('/home/pi/cnn_weather/cnn_weather_lite_quantized_edgetpu.tflite',
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    
    #load and arrange the test data
    data = pd.read_csv('/home/pi/cnn_weather/test.csv')
    test_data = np.asarray(data.loc[:len(data),"平均気温"],dtype=np.uint8)
    test_data = test_data.reshape(1,30,1)
    
    #measure time to predict
    start = time.perf_counter()
    
    #predict
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'],test_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    end = time.perf_counter()
    
    #print the output
    print("The next day's temperature is " + str(output_data[0,0]) + " degrees Celsius.")
    print("It took " + str((end-start)*1000) + " ms.")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
