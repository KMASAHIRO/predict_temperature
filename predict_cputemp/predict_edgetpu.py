'''
Get the cpu temperature every second for 30 seconds,
and predict the next temperature using the quantized
tflite model for edgetpu.
Output is the predicted next temperature and the real
next temperature, seconds it took to run this code.

Environments
Raspbian 10 (Raspberry pi4)
tflite_runtime 2.1.0.post1
edgetpu_runtime 2.1.4
'''

import numpy as np
import tflite_runtime.interpreter as tflite
import time
import subprocess

def main(args):
    start = time.perf_counter()
    
    #interpret the quantized tflite model for edgetpu
    interpreter = tflite.Interpreter('/home/pi/cnn_weather/cnn_weather_lite_quantized_edgetpu.tflite',
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    
    #get the cpu temperature and print it
    data = list()
    for i in range(30):
        res = subprocess.run(['cat', '/sys/class/thermal/thermal_zone0/temp'],
        stdout=subprocess.PIPE)
        get_start = time.perf_counter()
        result = res.stdout.decode('utf-8')
        result = int(result)/1000
        data.append(result)
        print(result,end='℃ ')
        if (i+1)%10 == 0:
            print()
        get_end = time.perf_counter()
        get_time = get_end-get_start
        
        #sleep until a second passes
        if get_time < 1:
            time.sleep(1-get_time)
        else:
            print("Took " + str(get_time) + " s to get " +  str(i) + "'s temp.")
    
    #arrange the data and predict
    pre_start = time.perf_counter()
    np_data = np.asarray(data,dtype=np.uint8).reshape(1,30,1)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'],np_data)
    interpreter.invoke()
    
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    pre_end = time.perf_counter()
    pre_time = pre_end - pre_start
    
    #print the output
    if pre_time < 1:
        print("The cpu's temp will be " + str(pred[0,0]) + "℃ in " + 
        str(1-pre_time) + " s.")
        
        time.sleep(1-pre_time)
        res = subprocess.run(['cat', '/sys/class/thermal/thermal_zone0/temp'],
        stdout=subprocess.PIPE)
        result = res.stdout.decode('utf-8')
        result = int(result)/1000
        print("The cpu's temp is " + str(result) + "℃.")
    else:
        print("The cpu's temp must have been " + str(pred[0,0]) + "℃ " + 
        str(1-pre_time) + " s ago.")
    
    end = time.perf_counter()
    print("Took " + str(end-start) + " s to run this code.")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
