# Demo Python Script for CISS / Data Logging with CISS in different modes

# Copyright 2020 Bosch Connected Devices and Solutions GmbH
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
#    endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# - After using 2kHz streaming of the acceleration sensor the node will go to reset and therefore any existing handles to the COM port must be freed
# Noise sensor is currently not implemented. Available only via BLE.
# CISS should be reset when changing the mode. This is especially important after 2kHz mode.

import serial, signal, configparser, os, csv, time, sys, enum

CISSpythonScriptVersion = "v03.01.00"
iniFileLocation = 'CISS_Configuration.ini'
printInformation = True
printInformation_Conf = True
sensor_id_glbl = "dummy"
rawDataLogFileMaxEntry = 25000   # So many data rows will be in RawData logfile (csv). You can change it.
eventLogFileMaxEntry = 5000      # So many data rows will be in Event logfile (csv). You can change it.
timeAggregationLogFileMaxEntry = 20000  # So many data rows will be in TimeAggregation logfile (csv). You can change it.

scriptPath = os.path.dirname(os.path.abspath(sys.argv[0]))
if not os.path.exists(scriptPath + "/RawData_LogFiles"): os.makedirs(scriptPath +"/RawData_LogFiles")
if not os.path.exists(scriptPath + "/Event_LogFiles"): os.makedirs(scriptPath +"/Event_LogFiles")
if not os.path.exists(scriptPath + "/TimeAggregation_LogFiles"): os.makedirs(scriptPath +"/TimeAggregation_LogFiles")

# helper for signed 16bit conversion
def s16(value):
    return -(value & 0x8000) | (value & 0x7fff)

# helper for ini file parsing
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#configure acc range    
def config_AccelerometerRange(ser, node):
    allowed_AccelerometerRanges = [2,4,8,16]
    #only allow valid AccelerometerRanges
    if node.AccelerometerRange in allowed_AccelerometerRanges:
        if printInformation_Conf: print('Set Accelerometer range to ' + str(node.AccelerometerRange) + 'g')
        conf_buff = bytearray([0xfe, 0x03, 0x80, 0x04, node.AccelerometerRange])
        write_conf(ser, conf_buff)
    else:
        node.AccelerometerRange = 16 #default
        if printInformation_Conf:
            print('Invalid Accelerometer Range given. CISS will operate with default (16g) or last valid value')


# crc calculation
def calc_crc(c_buff):
    result = 0
    for i in c_buff:
        result = result ^ i
    result = result ^ 254
    return result

# writes a string to the serial port with the possibility to
# have some debug output to stdout.
def write_conf(ser, conf_buf):   # send a command to the CISS
    crcByte = calc_crc(conf_buf)
    conf_buf.append(crcByte)
    bytes_send = bytes(conf_buf)
    conf_string = str(bytes_send)

    action = str(hex(bytes_send[2]))
    temp = hex(bytes_send[2])

    if temp == '0x80':
        action = "Accelerometer sensor"
    elif temp == '0x81':
        action = "Magnetometer sensor"
    elif temp == '0x82':
        action = "Gyroscope sensor"
    elif temp == '0x83':
        action = "Environmental sensors"
    elif temp == '0x84':
        action = "Light sensor"
    elif temp == '0x90':
        action = "BLE Chip"
    elif temp == '0x91':
        action = "Timestamp Setting"
    elif temp == '0xFC' or temp == '0xfc':
        action = "Event Detection Mode"
    elif temp == '0xFD' or temp == '0xfd':
        action = "Time Aggregation Mode"

    if printInformation_Conf: print(action, "command sent / command in hex: ", end = ''),
    for el in conf_string:
#        if printInformation_Conf: print(hex(ord(el))),
         if printInformation_Conf: print(el, end = '')
    if printInformation_Conf: print ('')

    # sent the command to the CISS
    ser.write(bytes_send)
    time.sleep(0.300)						 # delay between 2 Commands to the CISS device
    
# simple helper functions to parse the content of the payload
def parse_inert_vec(data):
    x = s16(data[0] | (data[1]<<8))
    y = s16(data[2] | (data[3]<<8))
    z = s16(data[4] | (data[5]<<8))
    array = [x, y, z]
    return array

def parse_temp(data):
    temp = s16(data[0] | (data[1]<<8))
    temp = float(temp)/10.0
    array = [temp]
    return array

def parse_press(data):
    press = data[0] | (data[1]<<8) | (data[2]<<16) | (data[3]<<24)
    press = float(press)/100
    array = [press]
    return array

def parse_humy(data):
    hum = s16(data[0] | (data[1]<<8))
    hum = float(hum)/100
    array = [hum]
    return array

def parse_light(data):
    light = data[0] | (data[1]<<8) | (data[2]<<16) | (data[3]<<24)
    array = [light]
    return array

def parse_aqu(data):
    aq = data[0] | (data[1]<<8)
    array = [aq]
    return array

def parse_enable(data):
    action = str(hex(data[0]))
    temp = hex(data[0])

    if temp == '0x80':
        action = "Accelerometer sensor"
    elif temp == '0x81':
        action = "Magnetometer sensor"
    elif temp == '0x82':
        action = "Gyroscope sensor"
    elif temp == '0x83':
        action = "Environmental sensors"
    elif temp == '0x84':
        action = "Light sensor"
    elif temp == '0x90':
        action = "BLE Chip"
    elif temp == '0x91':
        action = "Timestamp Setting"
    elif temp == '0xFC' or temp == '0xfc':
        action = "Event Detection Mode"
    elif temp == '0xFD' or temp == '0xfd':
        action = "Time Aggregation Mode"

    #if printInformation: print(action, "/ enable / command return hex:", hex(data[1]))
    return []

def parse_disable(data):
    action = str(hex(data[0]))
    temp = hex(data[0])

    if temp == '0x80':
        action = "Accelerometer sensor"
    elif temp == '0x81':
        action = "Magnetometer sensor"
    elif temp == '0x82':
        action = "Gyroscope sensor"
    elif temp == '0x83':
        action = "Environmental sensors"
    elif temp == '0x84':
        action = "Light sensor"
    elif temp == '0x90':
        action = "BLE Chip"
    elif temp == '0x91':
        action = "Timestamp Setting"
    elif temp == '0xFC' or temp == '0xfc':
        action = "Event Detection Mode"
    elif temp == '0xFD':
        action = "Time Aggregation Mode"

    if printInformation:
        print(action, "/ disable / command return hex:", hex(data[1]))
    return []

def parse_time_aggregation_inertial(data):
    axmin = float (s16(data[0] | (data[1]<<8)))
    axmax = float (s16(data[2] | (data[3]<<8)))
    axmean = float (s16(data[4] | (data[5]<<8)))
    axstd = float (s16(data[6] | (data[7]<<8)))
    aymin = float (s16(data[8] | (data[9]<<8)))
    aymax = float (s16(data[10] | (data[11]<<8)))
    aymean = float (s16(data[12] | (data[13]<<8)))
    aystd = float (s16(data[14] | (data[15]<<8)))
    azmin = float (s16(data[16] | (data[17]<<8)))
    azmax = float (s16(data[18] | (data[19]<<8)))
    azmean = float (s16(data[20] | (data[21]<<8)))
    azstd = float (s16(data[22] | (data[23]<<8)))
    amin = float (s16(data[24] | (data[25]<<8)))
    amax = float (s16(data[26] | (data[27]<<8)))
    amean = float (s16(data[28] | (data[29]<<8)))
    astd = float (s16(data[30] | (data[31]<<8)))

    gxmin = float (s16(data[32] | (data[33]<<8)))
    gxmax = float (s16(data[34] | (data[35]<<8)))
    gxmean = float (s16(data[36] | (data[37]<<8)))
    gxstd = float (s16(data[38] | (data[39]<<8)))
    gymin = float (s16(data[40] | (data[41]<<8)))
    gymax = float (s16(data[42] | (data[43]<<8)))
    gymean = float (s16(data[44] | (data[45]<<8)))
    gystd = float (s16(data[46] | (data[47]<<8)))
    gzmin = float (s16(data[48] | (data[49]<<8)))
    gzmax = float (s16(data[50] | (data[51]<<8)))
    gzmean = float (s16(data[52] | (data[53]<<8)))
    gzstd = float (s16(data[54] | (data[55]<<8)))
    gmin = float (s16(data[56] | (data[57]<<8)))
    gmax = float (s16(data[58] | (data[59]<<8)))
    gmean = float (s16(data[60] | (data[61]<<8)))
    gstd = float (s16(data[62] | (data[63]<<8)))

    if printInformation:
        print("Accelerometer x-axis [mg] ==> min:", axmin,", max:", axmax,", mean:", axmean,", std:", axstd)
        print("Accelerometer y-axis [mg] ==> min:", aymin,", max:", aymax,", mean:", aymean,", std:", aystd)
        print("Accelerometer z-axis [mg] ==> min:", azmin,", max:", azmax,", mean:", azmean,", std:", azstd)
        print("Accelerometer combined x,y,z [mg] ==> min:", amin,", max:", amax,", mean:", amean,", std:", astd)
        print("Gyroscope x-axis [°/s] ==> min:", gxmin,", max:", gxmax,", mean:", gxmean,", std:", gxstd)
        print("Gyroscope y-axis [°/s] ==> min:", gymin,", max:", gymax,", mean:", gymean,", std:", gystd)
        print("Gyroscope z-axis [°/s] ==> min:", gzmin,", max:", gzmax,", mean:", gzmean,", std:", gzstd)
        print("Gyroscope combined x,y,z [°/s] ==> min:", gmin,", max:", gmax,", mean:", gmean,", std:", gstd)

    array = [axmin,axmax,axmean,axstd,aymin,aymax,aymean,aystd,azmin,azmax,azmean,azstd,amin,amax,amean,astd,gxmin,gxmax,gxmean,gxstd,gymin,gymax,gymean,gystd,gzmin,gzmax,gzmean,gzstd,gmin,gmax,gmean,gstd ]
    write_to_csv_timeaggregation(sensor_id_glbl, 0x7E, array, int(time.time()*1000))
    return []

def parse_time_aggregation_temperature(data):
    tmin = s16(data[0] | (data[1]<<8))
    tmax = s16(data[2] | (data[3]<<8))
    tmean = s16(data[4] | (data[5]<<8))
    tstd = s16(data[6] | (data[7]<<8))
    tmin = float(tmin)/10
    tmax = float(tmax)/10
    tmean = float(tmean)/10
    tstd = float(tstd)/10
    array = [tmin, tmax, tmean, tstd]
    if (printInformation):
        print("Temperature [°C] - min:", tmin, ", max:", tmax, ", mean:", tmean, ", std:", tstd)
    write_to_csv_timeaggregation(sensor_id_glbl, 0x7D, array, int(time.time()*1000))
    return []


def parse_event_detection(data):
    global sensor_id_glbl
    # Event detection frames are parsed here
    print ("====== Detected Event: ======")
    print (' '.join([hex(i) for i in data[0:2]]))
    DeviceName = sensor_id_glbl
    Accel = data[0] & 0x03
    Gyro  = (data[0] & 0x0c) >> 2
    Mag   = (data[0] & 0X30) >> 4
    Temp  = (data[0] & 0xc0) >> 6
    Hum   = data[1] & 0x03
    Pressure = (data[1] & 0x0c) >> 2
    Light = (data[1] & 0x30) >> 4
    Noise = (data[1] & 0xc0) >> 6
    if Accel == 1:
        write_to_csv_event(int(time.time()*1000), "Accelerometer overshoot", str(node.eventlist["acc"].event_threshold[0]))
        if printInformation: print ("Accelerometer overshoot")
    if Accel == 3:
        write_to_csv_event(int(time.time()*1000), "Accelerometer undershoot", str(node.eventlist["acc"].event_threshold[0]))
        if printInformation: print ("Accelerometer undershoot")
    if Gyro == 1:
        write_to_csv_event(int(time.time()*1000), "Gyroscope overshoot", str(node.eventlist["gyr"].event_threshold[0]))
        if printInformation: print ("Gyroscope overshoot")
    if Gyro == 3:
        write_to_csv_event(int(time.time()*1000), "Gyroscope undershoot", str(node.eventlist["gyr"].event_threshold[0]))
        if printInformation: print ("Gyroscope undershoot")
    if Mag == 1:
        write_to_csv_event(int(time.time()*1000), "Magnetometer overshoot", str(node.eventlist["mag"].event_threshold[0]))
        if printInformation: print ("Magnetometer overshoot")
    if Mag == 3:
        write_to_csv_event(int(time.time()*1000), "Magnetometer undershoot", str(node.eventlist["mag"].event_threshold[0]))
        if printInformation: print ("Magnetometer undershoot")
    if Temp == 1:
        write_to_csv_event(int(time.time()*1000), "Temperature overshoot", str(node.eventlist["env"].event_threshold[0]))
        if printInformation: print ("Temperature overshoot")
    if Temp == 3:
        write_to_csv_event(int(time.time()*1000), "Temperature undershoot", str(node.eventlist["env"].event_threshold[0]))
        if printInformation: print ("Temperature undershoot")
    if Hum == 1:
        write_to_csv_event(int(time.time()*1000), "Humidity overshoot", str(node.eventlist["env"].event_threshold[1]))
        if printInformation: print ("Humidity overshoot")
    if Hum == 3:
        write_to_csv_event(int(time.time()*1000), "Humidity undershoot", str(node.eventlist["env"].event_threshold[1]))
        if printInformation: print ("Humidity undershoot")
    if Pressure == 1:
        write_to_csv_event(int(time.time()*1000), "Pressure overshoot", str(node.eventlist["env"].event_threshold[2]))
        if printInformation: print ("Pressure overshoot")
    if Pressure == 3:
        write_to_csv_event(int(time.time()*1000), "Pressure undershoot", str(node.eventlist["env"].event_threshold[2]))
        if printInformation: print ("Pressure undershoot")
    if Light == 1:
        write_to_csv_event(int(time.time()*1000), "Light overshoot", str(node.eventlist["light"].event_threshold[0]))
        if printInformation: print ("Light overshoot")
    if Light == 3:
        write_to_csv_event(int(time.time()*1000), "Light undershoot", str(node.eventlist["light"].event_threshold[0]))
        if printInformation: print ("Light undershoot")
    if Noise == 1:
        write_to_csv_event(int(time.time()*1000), "Noise overshoot", str(node.eventlist["noise"].event_threshold[0]))
        if printInformation: print ("Noise overshoot")
    if Noise == 3:
        write_to_csv_event(int(time.time()*1000), "Noise undershoot", str(node.eventlist["noise"].event_threshold[0]))
        if printInformation: print ("Noise undershoot")
    data = data[2:]

    return []



# Very simple class as a container for each sensor with its id in the payload stream data_idx
# and the corresponding length of the subsequent data (e.g. 6 bytes for 2b xyz-vector)
class Sensor:
    def __init__(self, t, d, p, a, b):
        self.data_idx = t
        self.data_length = d
        self.parser = p
        self.parse_begin = a
        self.parse_end = b

    def parse(self, data):
        res = self.parser(data)
        if res == []:
            return []

        mask = ['', '', '', '', '', '', '', '', '', '', '', '', '', '']
        for i in range(self.parse_begin, self.parse_end):
            mask[i] = res[i - self.parse_begin]
        return mask

# Configuration of the different streaming modes. This is separated from the sensor class
# because the environmental information is clustered and not every parameter can be configured
# individually for each sensor.
class StreamingConfig:
    def __init__(self, t, i, e, p, c, l):
        # environmental or inertial streaming
        self.streaming_type = t
        self.streaming_id = i
        self.streaming_enabled = e
        self.streaming_period = p
        self.cfg_id = c
        self.cfg_length = l

    def enable(self, ser, flag = True):
        conf_buff = bytearray([0xfe, 2])
        conf_buff.append(self.streaming_id)
        if flag: conf_buff.append(1)
        else: conf_buff.append(0)
        write_conf(ser, conf_buff)

    def disable(self, ser):
        self.enable(ser, False)

    def configure(self, ser, flgSetSamplingRateOnly):
        if ((self.streaming_enabled or flgSetSamplingRateOnly) and self.streaming_period>0):
            if printInformation_Conf: print("Set streaming period for", self.streaming_type, self.streaming_period)
                       
            conf_buff = bytearray([0xfe, self.cfg_length])
            conf_buff.append(self.cfg_id)
            conf_buff.append(2)
            k=0
            while ((len(conf_buff)-2) < self.cfg_length):
                conf_buff.append(int(self.streaming_period) >> (k*8) & 0xff)
                k+=1

            write_conf(ser, conf_buff)
        if (self.streaming_enabled and (self.streaming_period>0) and (flgSetSamplingRateOnly==0)):
            if printInformation_Conf: print("Enabling the sensor", self.streaming_type)
            self.enable(ser)
        
    

# Configuration of the different event modes. This is separated from the sensor class
# because the event information can be configured individually for each sensor
class EventConfig:
    def __init__(self, t, i, e, evt, l, cmd):
        self.event_type = t
        self.event_id = i
        self.event_enabled = e
        self.event_threshold = evt
        self.cfg_length = l
        self.event_cmd = cmd
        self.enabled = 0

    def enable(self, ser, flag = True):
        conf_buff = bytearray([0xfe, 0x02, 0xfc])
        if flag: conf_buff.append(1)
        else: conf_buff.append(0)
        write_conf(ser, conf_buff)

    def disable(self, ser):
        self.enable(ser, False)

    def configure(self, ser):
        if self.event_enabled:
            self.enabled = 1

            # create command bytes-----------------------
            for i in range(len(self.event_threshold)):
                if printInformation_Conf: print ("configure",self.event_type,"with threshold:",self.event_threshold[i])
                # set Event detection threshold
                conf_buff = bytearray([0xfe, (self.cfg_length[i] +2), self.event_id, self.event_cmd[i]])
                if (self.event_cmd[i] == 7): #handle temperature threshold separately, as value can be negative
                    if (self.event_threshold[i] > 127):
                        conf_buff.append(127)
                        if printInformation_Conf: print('Temperature limited to 127' )
                    elif (self.event_threshold[i] < -128):
                        conf_buff.append(128)
                        if printInformation_Conf: print('Temperature limited to -128' )
                    else:
                        conf_buff.append((256 + self.event_threshold[i])%256)            
                else:    
                    for j in range(self.cfg_length[i]):
                       conf_buff.append((int(self.event_threshold[i]) >> (j*8)) & 0xff)
                write_conf(ser, conf_buff)





def check_payload(payload):
    eval = 0
    for ind in range(len(payload)-1):
        eval = eval ^ payload[ind]

    if eval == payload[len(payload)-1]:
        return 1
    else:
        return 0

def conv_data(data):
    a = []
    b = data.decode('utf-16', 'ignore')
    c = list(b)
    for ind in range(len(c)):
        a.insert(ind, ord(c[ind]))
    return a

# Simple FileName Change to have every n time a new csv File
def getTimeStringForNewFile():	
	return time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime())

ts_count = 0
ts_diff = 0

TimeAggregation_Mode_Log_FilePath = ""
timeaggregationDataLineCounter = 0
def write_to_csv_timeaggregation(deviceName, sensorType, buff, tstamp):
    global ts_count, ts_diff, TimeAggregation_Mode_Log_FilePath, timeaggregationDataLineCounter

    if len(buff) < 2:
        return
    ts_diff = tstamp - ts_count

    if TimeAggregation_Mode_Log_FilePath == "":
        TimeAggregation_Mode_Log_FilePath = "./TimeAggregation_LogFiles/" + 'CISS_TimeAggregation_LogFile_' +deviceName+'_'+ getTimeStringForNewFile() + '.csv'

    if not os.path.exists(TimeAggregation_Mode_Log_FilePath):
        with open(TimeAggregation_Mode_Log_FilePath, 'w', newline = '' ) as csvOpen:
            csvobj = csv.writer(csvOpen, dialect='excel')
            csvobj.writerow(["sep=,"])
            csvobj.writerow(["Bosch Connected Devices and Solutions"])
            csvobj.writerow(["CISS Time Aggregation Log File in csv Format, Generated by: CISS Python Scripts " + CISSpythonScriptVersion])
            csvobj.writerow(["Explanation: In this log file the measurement values for accelerometer, gyroscope and temperature are aggregated in time frames and minimum, maximum, mean and standard deviation of the values are reported."])
            csvobj.writerow(["Accelerometer aggregation Time Frame: 2000ms (2 seconds)"])
            csvobj.writerow(["Accelerometer Measurement Range: ±"+ str(node.AccelerometerRange) +"g"])
            csvobj.writerow(["Accelerometer Sampling Rate: 100Hz (10ms)"])
            csvobj.writerow(["Accelerometer Unit: mg"])
            csvobj.writerow(["Gyroscope aggregation Time Frame: 2000ms (2 seconds)"])
            csvobj.writerow(["Gyroscope Measurement Range: ±2000°/s"])
            csvobj.writerow(["Gyroscope Sampling Rate: 100Hz (10ms)"])
            csvobj.writerow(["Gyroscope Unit: Degree/Second (°/s)"])
            csvobj.writerow(["Temperature aggregation Time Frame: 10000ms (10 seconds)"])
            csvobj.writerow(["Temperature Measurement Range: -20ºC - +80°C"])
            csvobj.writerow(["Temperature Sampling Rate: 1Hz (1000ms)"])
            csvobj.writerow(["Temperature Units: Degree Celcius (°C)"])
            csvobj.writerow([" "])
            csvobj.writerow(["Unix TimeStamp (ms)", "Measurement Type", "x_axis / min", "x_axis / max", "x_axis / mean", "x_axis / std", "y_axis / min", "y_axis / max", "y_axis / mean", "y_axis / std", "z_axis / min", "z_axis / max", "z_axis / mean", "z_axis / std", "combined_3_axis / min", "combined_3_axis / max", "combined_3_axis / mean", "combined_3_axis / std"])
            csvOpen.close()

    with open(TimeAggregation_Mode_Log_FilePath, "a", newline = '') as csvOpen:
        csvobj = csv.writer(csvOpen, dialect='excel')

        if (sensorType == 0x7E):  #Aggregated data for accelerometer and gyroscope
            if buff[0] != '':
                csvobj.writerow([tstamp, "Accelerometer", buff[0], buff[1], buff[2], buff[3], buff[4], buff[5], buff[6], buff[7], buff[8], buff[9], buff[10], buff[11], buff[12], buff[13], buff[14], buff[15]])
                timeaggregationDataLineCounter += 1
            if buff[16] != '':
                csvobj.writerow([tstamp, "Gyroscope", buff[16], buff[17], buff[18], buff[19], buff[20], buff[21], buff[22], buff[23], buff[24], buff[25], buff[26], buff[27], buff[28], buff[29], buff[30], buff[31]])
                timeaggregationDataLineCounter += 1

        if (sensorType == 0x7D):  # Aggregated data for Temperature
            if buff[0] != '':
                csvobj.writerow([tstamp, "Temperature", buff[0], buff[1], buff[2], buff[3]])
                timeaggregationDataLineCounter += 1

        if timeaggregationDataLineCounter >= timeAggregationLogFileMaxEntry:
            csvobj.writerow(["-----------------------------End of Log File - Totally " + str(timeaggregationDataLineCounter) + " Entries -----------------------------------"])
            TimeAggregation_Mode_Log_FilePath = ""
            timeaggregationDataLineCounter = 0
        csvOpen.close()



rawDataLogFilePath = ""
rawDataLineCounter = 0
# simple helper to write sensor data to a csv file
def write_to_csv(self, buff, tstamp):
    global ts_count, ts_diff, rawDataLogFilePath, rawDataLineCounter, rawDataLogFileMaxEntry

    if len(buff) < 14:
        return
    ts_diff = tstamp - ts_count

    if rawDataLogFilePath == "":
        rawDataLogFilePath = "./RawData_LogFiles/" + 'CISS_RawData_LogFile_' + self.DeviceName + '_' + getTimeStringForNewFile() + '.csv'

    if not os.path.exists(rawDataLogFilePath):
        with open(rawDataLogFilePath, 'w', newline = '' ) as csvOpen:
            csvobj = csv.writer(csvOpen, dialect='excel')
            csvobj.writerow(["sep=,"])
            csvobj.writerow(["Bosch Connected Devices and Solutions"])
            csvobj.writerow(["CISS Raw Data Log File in csv Format, Generated by: CISS Python Scripts "+ CISSpythonScriptVersion])
            csvobj.writerow(["Explanation: In this log file the measurement values (raw-data) from the sensors are reported."])
            csvobj.writerow(["Accelerometer Measurement Range: ±"+ str(node.AccelerometerRange) +"g"])
            inertialSamplingRate = str(1000000/node.streaminglist["acc"].streaming_period) + "Hz (" + str(node.streaminglist["acc"].streaming_period/1000) +"ms)"
            if self.streaminglist["acc"].streaming_enabled:
                csvobj.writerow(["Accelerometer Sampling Rate: "+ inertialSamplingRate ])
            else:
                csvobj.writerow(["Accelerometer Sampling Rate: disabled"])
            csvobj.writerow(["Accelerometer Unit: mg"])
            csvobj.writerow(["Magnetometer Measurement Range: ±1300µT (x and y-Axis); ±2500µT (z-Axis)"])
            if self.streaminglist["mag"].streaming_enabled:
                csvobj.writerow(["Magnetometer Sampling Rate: "+ inertialSamplingRate ])
            else:
                csvobj.writerow(["Magnetometer Sampling Rate: disabled"])
            csvobj.writerow(["Magnetometer Unit: Micro Tesla (µT)"])
            csvobj.writerow(["Gyroscope Measurement Range: ±2000°/s"])
            if self.streaminglist["gyr"].streaming_enabled:
                csvobj.writerow(["Gyroscope Sampling Rate: "+ inertialSamplingRate ])
            else:
                csvobj.writerow(["Gyroscope Sampling Rate: disabled"])
            csvobj.writerow(["Gyroscope Unit: Degree/Second (°/s)"])
            environmentSamplingRate = str(node.streaminglist["env"].streaming_period) + "Hz (" + str(node.streaminglist["env"].streaming_period * 1000) + "ms)"
            csvobj.writerow(["Light Measurement Range: 0 - 2112800 lux"])
            if self.streaminglist["light"].streaming_enabled:
                csvobj.writerow(["Light Sampling Rate: "+environmentSamplingRate])
            else:
                csvobj.writerow(["Light Sampling Rate: disabled"])
            csvobj.writerow(["Light Unit: lux"])
            csvobj.writerow(["Temperature Measurement Range: -20ºC – +80°C"])
            if self.streaminglist["env"].streaming_enabled:
                csvobj.writerow(["Temperature Sampling Rate: "+environmentSamplingRate])
            else:
                csvobj.writerow(["Temperature Sampling Rate: disabled"])
            csvobj.writerow(["Temperature Units: Degree Celcius (°C)"])
            csvobj.writerow(["Humidity Measurement Range: 20 - 90% rH (non-condensing)"])
            if self.streaminglist["env"].streaming_enabled:
                csvobj.writerow(["Humidity Sampling Rate: "+environmentSamplingRate])
            else:
                csvobj.writerow(["Humidity Sampling Rate: disabled"])
            csvobj.writerow(["Humidity Units: % Relative Humidity (rH)"])
            csvobj.writerow(["Pressure Measurement Range: 300 - 1100 hPa"])
            if self.streaminglist["env"].streaming_enabled:
                csvobj.writerow(["Pressure Sampling Rate: "+environmentSamplingRate])
            else:
                csvobj.writerow(["Pressure Sampling Rate: disabled"])
            csvobj.writerow(["Pressure Units: hPa"])
            csvobj.writerow(["Noise/Acustic Measurement Range: 0 - 4096 (not available via USB, available only via BLE)"])
            csvobj.writerow(["Noise/Acustic Sampling Rate: disabled"])
            csvobj.writerow(["Noise/Acustic Units: no unit (0-100% with 0-4096 range)"])
            csvobj.writerow([" "])
            csvobj.writerow(["Unix TimeStamp (ms)", "Measurement Type", "Data / x_axis", "y_axis", "z_axis"])
            csvOpen.close()
        firstTimeStamp = tstamp

    with open(rawDataLogFilePath, "a", newline = '') as csvOpen:
        csvobj = csv.writer(csvOpen, dialect='excel')

        if buff[0] != '':
            csvobj.writerow([tstamp, "Accelerometer", buff[0], buff[1], buff[2]])
        elif buff[3] != '':
            csvobj.writerow([tstamp, "Gyroscope", buff[3], buff[4], buff[5]])
        elif buff[6] != '':
            csvobj.writerow([tstamp, "Magnetometer", buff[6], buff[7], buff[8]])
        elif buff[9] != '':
            csvobj.writerow([tstamp, "Temperature", buff[9]])
        elif buff[10] != '':
            csvobj.writerow([tstamp, "Pressure", buff[10]])
        elif buff[11] != '':
            csvobj.writerow([tstamp, "Humidity", buff[11]])
        elif buff[12] != '':
            csvobj.writerow([tstamp, "Light", buff[12]])
        else:
            csvobj.writerow([tstamp, "Error in data string."])


        rawDataLineCounter += 1
        if rawDataLineCounter >= rawDataLogFileMaxEntry:
            csvobj.writerow(["-----------------------------End of Log File - Totally " + str(rawDataLineCounter) + " Entries -----------------------------------"])
            rawDataLogFilePath = ""
            rawDataLineCounter = 0
            csvOpen.close()

        # write streaming data to the screen with 100ms = 10Hz -----------------------------
        if ts_diff > 100:
            if printInformation:
                if buff[0] != '':
                    print("TimeStamp:", tstamp, "- Accelerometer [mg]", "x:", buff[0], "y:", buff[1], "z:", buff[2])
                elif buff[3] != '':
                    print("TimeStamp:", tstamp, "- Gyroscope [°/s]", "x:", buff[3], "y:", buff[4], "z:", buff[5])
                elif buff[6] != '':
                    print("TimeStamp:", tstamp, "- Magnetometer [µT]", "x:", buff[6], "y:", buff[7], "z:", buff[8])
                elif buff[9] != '':
                    print("TimeStamp:", tstamp, "- Temperature [°C]:", buff[9])
                elif buff[10] != '':
                    print("TimeStamp:", tstamp, "- Pressure [hPa]:", buff[10])
                elif buff[11] != '':
                    print("TimeStamp:", tstamp, "- Humidity [%rH]:", buff[11])
                elif buff[12] != '':
                    print("TimeStamp:", tstamp, "- Light [Lux]:", buff[12])
                else:
                    print("Error in data string.")
            ts_count = tstamp


file = None
fileObject = None
def is_new_file_needed():
    if rawDataLogFilePath == "":
        return True
    else:
        return False


def createNew2kHzLogFile(self, tstamp):
    global rawDataLogFilePath, firstTimeStamp

    rawDataLogFilePath = "./RawData_LogFiles/" + 'CISS_RawData_LogFile_' + self.DeviceName + '_' + getTimeStringForNewFile() + '.csv'

    if not os.path.exists(rawDataLogFilePath):
        with open(rawDataLogFilePath, 'w', newline='') as csvOpen:
            csvobj = csv.writer(csvOpen, dialect='excel')
            csvobj.writerow(["sep=,"])
            csvobj.writerow(["Bosch Connected Devices and Solutions"])
            csvobj.writerow(
                ["CISS Raw Data Log File in csv Format, Generated by: CISS Python Scripts " + CISSpythonScriptVersion])
            csvobj.writerow(["Special Mode: This is 2kHz_Accelerometer_Mode. All the other sensors are disabled."])
            csvobj.writerow(["Explanation: In this log file the measurement values (raw-data) from the accelerometer sensor are reported."])
            csvobj.writerow(["Accelerometer Measurement Range: ±" + str(node.AccelerometerRange) + "g"])
            inertialSamplingRate = str(1000000 / node.streaminglist["acc"].streaming_period) + "Hz (" + str(
                node.streaminglist["acc"].streaming_period / 1000) + "ms)"
            csvobj.writerow(["Accelerometer Sampling Rate: " + inertialSamplingRate])
            csvobj.writerow(["Accelerometer Unit: mg"])
            csvobj.writerow([" "])
            csvobj.writerow(["Unix TimeStamp (ms)", "Measurement Type", "Data / x_axis", "y_axis", "z_axis"])
            csvOpen.close()
        firstTimeStamp = tstamp


def keep_log_file_open():
    global file, fileObject

    try:
        file = open(rawDataLogFilePath, "a", newline='')
        fileObject = csv.writer(file, dialect='excel')
    except:
        print("Unable to open 2KHz log file.")


def write_2KHz_packets_to_file(buff, tstamp):
    global ts_count, ts_diff, rawDataLineCounter, fileObject

    if len(buff) >= 14:
        ts_diff = tstamp - ts_count

        # Write to log file
        if buff[0] != '':
            fileObject.writerow([tstamp, "Accelerometer", buff[0], buff[1], buff[2]])
        else:
            fileObject.writerow([tstamp, "Error in data string."])

        # Increment the data line counter
        rawDataLineCounter += 1

        # Display streaming data every 100ms = 10Hz -----------------------------
        if ts_diff > 100:
            if printInformation:
                if buff[0] != '':
                    print("TimeStamp:", tstamp, "- Accelerometer [mg]", "x:", buff[0], "y:", buff[1], "z:", buff[2])
                else:
                    print("Error in data string.")
            ts_count = tstamp


def is_2kHzfile_full():
    global rawDataLineCounter, rawDataLogFileMaxEntry

    if rawDataLineCounter >= rawDataLogFileMaxEntry:
        return True
    else:
        return False


def close_2kHz_file(tstamp):
    global firstTimeStamp, rawDataLineCounter, rawDataLogFilePath, file, fileObject

    real_kHz = rawDataLineCounter / (tstamp - firstTimeStamp)
    fileObject.writerow(["-----------------------------End of Log File - Totally " + str(
        rawDataLineCounter) + " Entries (Average: ~" + str(round(real_kHz, 2)) + " kHz)-----------------------------------"])
    rawDataLogFilePath = ""
    rawDataLineCounter = 0
    file.close()


eventDataLineCounter = 0
eventLogFilePath = ""
def write_to_csv_event(tstamp, event, threshold):
    global eventDataLineCounter, eventLogFilePath, eventLogFileMaxEntry
    if eventLogFilePath == "" or eventDataLineCounter >= eventLogFileMaxEntry:

        if eventLogFilePath == "":
            eventLogFilePath = "./Event_LogFiles/" + 'CISS_Event_LogFile_' + node.DeviceName + '_' + getTimeStringForNewFile() + '.csv'

    if not os.path.exists(eventLogFilePath):
        with open(eventLogFilePath, "w", newline = '') as csvOpen:
            csvobj = csv.writer(csvOpen, dialect='excel')
            # Add event log file header information--------------------
            csvobj.writerow(["sep=,"])
            csvobj.writerow(["Bosch Connected Devices and Solutions"])
            csvobj.writerow(["CISS Event Log File in csv Format, Generated by: CISS Python Scripts "+ CISSpythonScriptVersion])
            csvobj.writerow(["Explanation: In this log file the events from the threshold violations are reported."])
            csvobj.writerow(["Accelerometer Measurement Range: ±" + str(node.AccelerometerRange) + "g"])
            inertialSamplingRate = str(1000000 / node.streaminglist["acc"].streaming_period) + "Hz (" + str(
                node.streaminglist["acc"].streaming_period / 1000) + "ms)"
            csvobj.writerow(["Accelerometer Sampling Rate: " + inertialSamplingRate])
            csvobj.writerow(["Accelerometer Unit: mg"])
            csvobj.writerow(["Magnetometer Measurement Range: ±1300µT (x and y-Axis); ±2500µT (z-Axis)"])
            csvobj.writerow(["Magnetometer Sampling Rate: " + inertialSamplingRate])
            csvobj.writerow(["Magnetometer Unit: Micro Tesla (µT)"])
            csvobj.writerow(["Gyroscope Measurement Range: ±2000°/s"])
            csvobj.writerow(["Gyroscope Sampling Rate: " + inertialSamplingRate])
            csvobj.writerow(["Gyroscope Unit: Degree/Second (°/s)"])
            environmentSamplingRate = str(node.streaminglist["env"].streaming_period) + "Hz (" + str(
                node.streaminglist["env"].streaming_period * 1000) + "ms)"
            csvobj.writerow(["Light Measurement Range: 0 - 2112800 lux"])
            csvobj.writerow(["Light Sampling Rate: " + environmentSamplingRate])
            csvobj.writerow(["Light Unit: lux"])
            csvobj.writerow(["Temperature Measurement Range: -20ºC - +80°C"])
            csvobj.writerow(["Temperature Sampling Rate: " + environmentSamplingRate])
            csvobj.writerow(["Temperature Units: Degree Celcius (°C)"])
            csvobj.writerow(["Humidity Measurement Range: 20 - 90% rH (non-condensing)"])
            csvobj.writerow(["Humidity Sampling Rate: " + environmentSamplingRate])
            csvobj.writerow(["Humidity Units: % Relative Humidity (rH)"])
            csvobj.writerow(["Pressure Measurement Range: 300 - 1100 hPa"])
            csvobj.writerow(["Pressure Sampling Rate: " + environmentSamplingRate])
            csvobj.writerow(["Pressure Units: hPa"])
            csvobj.writerow(["Noise/Acustic Measurement Range: 0 - 4096 (not available via USB, available only via BLE)"])
            csvobj.writerow(["Noise/Acustic Sampling Rate: disabled"])
            csvobj.writerow(["Noise/Acustic Units: no unit (0-100% with 0-4096 range)"])
            csvobj.writerow([" "])
            csvobj.writerow(["Unix TimeStamp (ms)", "Event Type", "Threshold"])

    with open(eventLogFilePath, "a", newline = '' ) as csvOpen:
        csvobj = csv.writer(csvOpen, dialect='excel')
        csvobj.writerow([tstamp, event, threshold])

        eventDataLineCounter += 1

        if eventDataLineCounter >= eventLogFileMaxEntry:
            csvobj.writerow(["-----------------------------End of Log File - Totally " + str(eventDataLineCounter) + " Entries -----------------------------------"])
            eventLogFilePath = ""
            eventDataLineCounter = 0
        csvOpen.close()
           
out = 0



class Modes(enum.Enum):
   RawDataStreaming_Mode = 1
   Accelerometer_2KHz_Mode  = 2
   TimeAggregation_Mode  = 3
   EventDetection_Mode = 4

class CISSNode:
    def __init__(self):
        #initialize classes, dictionaries and variables
        self.flgEventEnabled = 0
        no_sens = Sensor(0, 0, parse_enable, 0, 0)
        enable = Sensor(1, 2, parse_enable, 0, 0)
        acc = Sensor(2, 6, parse_inert_vec, 0, 3)
        mag = Sensor(3, 6, parse_inert_vec, 6, 9)
        gyro = Sensor(4, 6, parse_inert_vec, 3, 6)
        temp = Sensor(5, 2, parse_temp, 9, 10)
        press = Sensor(6, 4, parse_press, 10, 11)
        humy = Sensor(7, 2, parse_humy, 11, 12)
        light = Sensor(8, 4, parse_light, 12, 13)
        aqu = Sensor(9, 2, parse_aqu, 13, 14)
        disable = Sensor(255, 2, parse_enable, 0, 0)
        event_detection = Sensor(0x7A, 2, parse_event_detection, 0, 0)
        TimeAggregation_Mode_inertial = Sensor(0x7E, 64, parse_time_aggregation_inertial, 0, 63)
        TimeAggregation_Mode_temperature = Sensor(0x7D, 8, parse_time_aggregation_temperature, 0, 7)

        self.sensorlist = [no_sens, acc, mag, gyro, temp, press, humy, light, aqu, enable, disable, event_detection, TimeAggregation_Mode_inertial, TimeAggregation_Mode_temperature]

        streaming_acc = StreamingConfig("inertial_acc", 0x80, True,   100000, 0x80, 6)
        streaming_gyr = StreamingConfig("inertial_gyr", 0x82, False,  100000, 0x82, 6)
        streaming_mag = StreamingConfig("inertial_mag", 0x81, False,  100000, 0x81, 6)
        streaming_env = StreamingConfig("env_TempHumidPress", 0x83, False, 1000000, 0x83, 4)
        # light sensor 
        streaming_light = StreamingConfig("env_light", 0x84, False, 1000000, 0x84, 4)
        self.streaminglist = {"env": streaming_env, 
                              "acc": streaming_acc,
                              "mag": streaming_mag, 
                              "gyr": streaming_gyr,
                              "light": streaming_light}
        event_acc = EventConfig("event_acc", 0x80, False, [0], [2], [3])
        event_gyr = EventConfig("event_gyr", 0x82, False, [0], [2], [3])
        event_mag = EventConfig("event_mag", 0x81, False, [0], [2], [3])
        event_env = EventConfig("event_env", 0x83, False, [0,0,0], [1,1,3], [7,8,9])
        event_noise = EventConfig("event_noise", 0x85, False, [0], [2], [3])
        event_light = EventConfig("event_light", 0x84, False, [0], [3], [3])
        self.eventlist = {"env": event_env, 
                          "acc": event_acc,
                          "mag": event_mag, 
                          "gyr": event_gyr,
                          "noise": event_noise,
                          "light": event_light }
        self.AccelerometerRange = 16
        self.Mode = Modes.RawDataStreaming_Mode

        # read values from the ini file
        self.get_ini_config()
        self.checkEventEnabled()

        # connect to CISS COM port
        self.connect()

        # disable all sensors before configuration
        self.disable_sensors()
        
        # configure the sensors as given in the CISS_Configuration.ini file
        self.config_sensors()

    def get_ini_config(self):
        global sensor_id_glbl
        global iniFileLocation
        global printInformation
        global printInformation_Conf

        if not os.path.exists(iniFileLocation):
            sIRConf = configparser.ConfigParser()
            sIRConf.add_section("CISS_configuration")
            sIRConf.set("CISS_configuration", "port", "/dev/ttyACM0")
            sIRConf.set("CISS_configuration", "DeviceName", "myCISS")
            sIRConf.set("CISS_configuration", "PrintToScreen", "true")
            sIRConf.set("CISS_configuration", "AccelerometerRange", "16")

            sIRConf.set("CISS_configuration", "InertialSamplingRate", "10")
            sIRConf.set("CISS_configuration", "EnvironmentalSamplingRate", "1000")

            sIRConf.set("CISS_configuration", "RawDataStreaming_Mode", "true")
            sIRConf.set("CISS_configuration", "Accelerometer_2KHz_Mode", "false")
            sIRConf.set("CISS_configuration", "TimeAggregation_Mode", "false")
            sIRConf.set("CISS_configuration", "EventDetection_Mode", "false")

            sIRConf.set("CISS_configuration", "AccelerometerThreshold", "4")
            sIRConf.set("CISS_configuration", "MagnetometerThreshold", "500")
            sIRConf.set("CISS_configuration", "GyroscopeThreshold", "55")
            sIRConf.set("CISS_configuration", "TemperatureThreshold", "15")
            sIRConf.set("CISS_configuration", "HumidityThreshold", "45")
            sIRConf.set("CISS_configuration", "PressureThreshold", "600")
            sIRConf.set("CISS_configuration", "LightThreshold", "350")

            with open(iniFileLocation, "w") as newcfgfile:
                sIRConf.write(newcfgfile)

        snIniConfig = configparser.ConfigParser()
        snIniConfig.read(iniFileLocation, encoding='utf-8-sig')

        self.port = snIniConfig.get("CISS_configuration", "port")
        self.DeviceName = snIniConfig.get("CISS_configuration", "DeviceName")
        sensor_id_glbl = self.DeviceName

        if str2bool(snIniConfig.get("CISS_configuration", "PrintToScreen")) == 0:
            printInformation = False

        InertialSamplingRate = int(snIniConfig.get("CISS_configuration", "InertialSamplingRate"))*1000
        EnvironmentalSamplingRate = int(snIniConfig.get("CISS_configuration", "EnvironmentalSamplingRate"))/1000
        self.AccelerometerRange = int(snIniConfig.get("CISS_configuration", "AccelerometerRange"))

        # Row-Data Streaming Mode (up to 100Hz)================================
        self.streaminglist["acc"].streaming_enabled = str2bool(snIniConfig.get("CISS_configuration", "Accelerometer_Streaming"))
        self.streaminglist["mag"].streaming_enabled = str2bool(snIniConfig.get("CISS_configuration", "Magnetometer_Streaming"))
        self.streaminglist["gyr"].streaming_enabled = str2bool(snIniConfig.get("CISS_configuration", "Gyroscope_Streaming"))
        self.streaminglist["env"].streaming_enabled = str2bool(snIniConfig.get("CISS_configuration", "Environmental_Streaming"))
        self.streaminglist["light"].streaming_enabled = str2bool(snIniConfig.get("CISS_configuration", "Light_Streaming"))

        # Mode selection-----------------
        if str2bool(snIniConfig.get("CISS_configuration", "RawDataStreaming_Mode")):
            self.Mode = Modes.RawDataStreaming_Mode
            print("\nCISS is started to configure for RawDataStreaming_Mode...")
        elif str2bool(snIniConfig.get("CISS_configuration", "Accelerometer_2KHz_Mode")):
            self.Mode = Modes.Accelerometer_2KHz_Mode
            print("\nCISS is started to configure for Accelerometer_2KHz_Mode...")
            InertialSamplingRate = 500   # 500 Microseconds
            self.streaminglist["acc"].streaming_enabled = True   # only accelerometer is active in 2kHz Mode
            self.streaminglist["mag"].streaming_enabled = False
            self.streaminglist["gyr"].streaming_enabled = False
            self.streaminglist["env"].streaming_enabled = False
            self.streaminglist["light"].streaming_enabled = False
            printInformation = False  # in 2kHz we don't write the values on the screen
        elif str2bool(snIniConfig.get("CISS_configuration", "TimeAggregation_Mode")):
            self.Mode = Modes.TimeAggregation_Mode
            print("\nCISS is started to configure for TimeAggregation_Mode...")
        elif str2bool(snIniConfig.get("CISS_configuration", "EventDetection_Mode")):
            self.Mode = Modes.EventDetection_Mode
            print("\nCISS is started to configure for EventDetection_Mode...")
        else:
            print("No valid mode selection in ini file. The default Mode RawDataStreaming_Mode is activated")
            print("\nCISS is started to configure for RawDataStreaming_Mode...")
            self.Mode = Modes.RawDataStreaming_Mode

        # set sampling rates------------
        self.streaminglist["env"].streaming_period = EnvironmentalSamplingRate
        self.streaminglist["light"].streaming_period = EnvironmentalSamplingRate
        self.streaminglist["acc"].streaming_period = InertialSamplingRate
        self.streaminglist["mag"].streaming_period = InertialSamplingRate
        self.streaminglist["gyr"].streaming_period = InertialSamplingRate



        # Event Detection Mode ================================================
        EventMode_enabled = 0
        if self.Mode == Modes.EventDetection_Mode:
            EventMode_enabled = 1

        self.eventlist["env"].event_enabled = EventMode_enabled
        self.eventlist["env"].event_threshold = [int(snIniConfig.get("CISS_configuration", "TemperatureThreshold")),
                                                 int(snIniConfig.get("CISS_configuration", "HumidityThreshold")),
                                                 int(snIniConfig.get("CISS_configuration", "PressureThreshold"))]
        self.eventlist["acc"].event_enabled = EventMode_enabled
        self.eventlist["acc"].event_threshold = [int(snIniConfig.get("CISS_configuration", "AccelerometerThreshold"))]
        self.eventlist["mag"].event_enabled = EventMode_enabled
        self.eventlist["mag"].event_threshold = [int(snIniConfig.get("CISS_configuration", "MagnetometerThreshold"))]
        self.eventlist["gyr"].event_enabled = EventMode_enabled
        self.eventlist["gyr"].event_threshold = [int(snIniConfig.get("CISS_configuration", "GyroscopeThreshold"))]
        self.eventlist["light"].event_enabled = EventMode_enabled
        self.eventlist["light"].event_threshold = [int(snIniConfig.get("CISS_configuration", "LightThreshold"))]


    def connect(self):
        self.ser = serial.Serial(self.port,baudrate=115200,timeout=1)

    def checkEventEnabled(self):
        for elem in self.eventlist.values():
            if elem.event_enabled == 1:
                self.flgEventEnabled = 1

    def disconnect(self):
        self.disable_sensors()
        self.ser.close()

    def disable_sensors(self):
        if printInformation_Conf: print("All Sensors and modes will be disabled before the configuration...")
        self.ser.flush()
        # Disable all enabled sensors exept for accel since it in the special mode of 2K streaming,
        # diabling the accel will trigger a node reset. Therefore, no serial port operations
        # will be possible until the node reboots and thus the accel disable should be sent the
        # last. This is introduced with v02.01.00
        #
        for elem in self.streaminglist.values():
            #if elem.streaming_enabled:
            if printInformation_Conf: print("Disabling the sensor: ", elem.streaming_type)
            elem.disable(self.ser)

        if (self.flgEventEnabled == 1):
            if printInformation_Conf: print("Disabling the Event Detection mode.")
            conf_buff = bytearray([0xfe, 0x02, 0xfc, 0x00])
            write_conf(self.ser, conf_buff)
			
        if (self.Mode == Modes.TimeAggregation_Mode):
            if printInformation_Conf: print("Disabling the TimeAggregation mode.")
            conf_buff = bytearray([0xfe, 0x02, 0xfd, 0x00])
            write_conf(self.ser, conf_buff)


            
    def enable_sensors(self):  # Enable the sensors for Raw-Data Streaming mode and Event Detection mode
        for elem in self.streaminglist.values():
            if elem.streaming_enabled:
                elem.enable(self.ser)

        for elem in self.eventlist.values():
            if elem.event_enabled:
                elem.enable(self.ser)


    def get_type(self, num):
        a = -1
        for i in range(len(self.sensorlist)):
            if num == self.sensorlist[i].data_idx:
                a = i
        return a



    def parse_payload(self, payload):
        payload.pop(0)
        payload.pop(len(payload)-1)
        while len(payload) != 0:
            t = self.get_type(payload[0])
            payload.pop(0)
            if t >= 0:
                tstamp = int(time.time()*1000)
                mask = self.sensorlist[t].parse(payload[0:self.sensorlist[t].data_length])
                payload = payload[self.sensorlist[t].data_length:]
                if (self.Mode == Modes.Accelerometer_2KHz_Mode):
                    if is_new_file_needed():
                        createNew2kHzLogFile(self, tstamp)
                        keep_log_file_open()
                    write_2KHz_packets_to_file(mask,tstamp)
                    if is_2kHzfile_full():
                        close_2kHz_file(tstamp)
                else:
                    write_to_csv(self, mask, int(time.time()*1000))
            else:
                break






    def stream(self):
        global out    
        sof = b'\xFE'
        data = []
        sub_payload = []
        payload_found = 0
        payload = []
        sr = self.ser
        while  payload_found != 1:
            while not out == sof:
                out = sr.read()
    
            length = sr.read()
            if length:
                length = ord(length)
            else:
                continue
            buffer = sr.read(length+1)
            payload = bytearray(buffer)
            payload.insert(0, length)
            out = ""
            if check_payload(payload) == 1:
                payload_found = 1
                self.parse_payload(payload)

    def config_sensors(self):
        if printInformation_Conf: print("\nConfiguration is started...")
        # Configure range of acc if acc streaming or threshold detection is enabled
        # As long as only acc sensor can has a configurable range this will not be
        # handled by a separate method of StreamingConfig or EventConfig 

        # Set the range of the Accelerometer
        if (self.eventlist["acc"].event_enabled or self.streaminglist["acc"].streaming_enabled):
            config_AccelerometerRange(self.ser, self)

        # Check if any event is activated in the ini file
        eventActivated = False
        #check if one of the inertial sensors is enabled in event detection mode, if yes set the sample period
        if (self.eventlist["acc"].event_enabled or self.eventlist["mag"].event_enabled or self.eventlist["gyr"].event_enabled):
            self.streaminglist["acc"].configure(self.ser, 1)
            eventActivated = True
        #check if env, light or noise is enabled in event detection mode, if yes set the sample period 
        if (self.eventlist["env"].event_enabled or self.eventlist["light"].event_enabled or self.eventlist["noise"].event_enabled):            
            self.streaminglist["env"].configure(self.ser, 1)
            eventActivated = True

        #################################################################################
        ## CISS Mode Activation #########################################################
        if self.Mode == Modes.TimeAggregation_Mode:
            print('Time Aggregation Mode is started. Please see the log files.')
            conf_buff = bytearray([0xfe, 0x02, 0xfd, 0x01])
            write_conf(self.ser, conf_buff)
        elif eventActivated:
            print('Event Mode is started. Please see the log files.')
            for elem in self.eventlist.values():
                elem.configure(self.ser)
            # send the start event detection mode only at the end of the event mode configuration (i.e. respective thresholds)
            self.eventlist["acc"].enable(self.ser)  # enabling of the event mode, therefore use "acc" instance,
        else:
            # configure and start the sensors for Raw-Data Streaming Mode
            for elem in self.streaminglist.values():
                elem.configure(self.ser, 0)


def ctrl_c_handler(signal, frame,node):
    raise Exception("")

node = CISSNode()

def main():
    signal.signal(signal.SIGINT, ctrl_c_handler)
    print("\nData Streaming is started. Please see the log files...")
    while 1:
        node.stream()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if printInformation: print("disconnected")
        node.disconnect()
        time.sleep(1)
        exit(0)
