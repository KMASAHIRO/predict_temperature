#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  predict_cpu.py
#  
#  Copyright 2020  <pi@raspberrypi>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import tflite_runtime.interpreter as tflite
import time
import subprocess

def main(args):
    start = time.perf_counter()
    interpreter = tflite.Interpreter('/home/pi/cnn_weather/cnn_weather_lite_quantized_edgetpu.tflite',
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    
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
        
        if get_time < 1:
            time.sleep(1-get_time)
        else:
            print("Took " + str(get_time) + " s to get " +  str(i) + "'s temp.")
    
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
    if pre_time < 1:
        print("The cpu's temp will be " + str(pred[0,0]) + " ℃ in " + 
        str(1-pre_time) + " s.")
    else:
        print("The cpu's temp was " + str(pred[0,0]) + " ℃ " + 
        str(1-pre_time) + " s ago.")
    
    end = time.perf_counter()
    print("Took " + str(end-start) + " s to run this code.")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
