#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cpu_quantized_time.py
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
import pandas as pd
import tflite_runtime.interpreter as tflite
import time 

def main(args):
    interpreter = tflite.Interpreter('/home/pi/cnn_weather/cnn_weather_lite_quantized.tflite')
    
    data = pd.read_csv('/home/pi/cnn_weather/test.csv')
    test_data = np.asarray(data.loc[:len(data),"平均気温"],dtype=np.uint8)
    test_data = test_data.reshape(1,30,1)
    
    start = time.perf_counter()
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'],test_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    end = time.perf_counter()
    
    print("The next day's temperature is " + str(output_data[0,0]) + " degrees Celsius.")
    print("It took " + str((end-start)*1000) + " ms.")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
