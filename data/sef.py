import numpy as np
import struct




def read( filename ):
    
    stream = open(filename,'rb')
    s = struct.Struct('4s')
    version = s.unpack(stream.read(4))[0]
    
    s = struct.Struct('i')
    number_channels = s.unpack(stream.read(4))[0]
    number_aux_channels = s.unpack(stream.read(4))[0]
    number_timeframes = s.unpack(stream.read(4))[0]

    s = struct.Struct('f')
    samplingrate = s.unpack(stream.read(4))[0]

    s = struct.Struct('h')
    year = s.unpack(stream.read(2))[0]
    month = s.unpack(stream.read(2))[0]
    day = s.unpack(stream.read(2))[0]
    hour = s.unpack(stream.read(2))[0]
    minute = s.unpack(stream.read(2))[0]
    second = s.unpack(stream.read(2))[0]
    millisecond = s.unpack(stream.read(2))[0]
    
    info = []
    info.append(("version:",version))
    info.append(("number of channels:",number_channels))
    info.append(("number of aux channels:",number_aux_channels))
    info.append(("number of timeframes:",number_timeframes))
    info.append(("samplingrate:",samplingrate))
    info.append(("year:",year))
    info.append(("month:",month))
    info.append(("day:",day))
    info.append(("hour:",hour))
    info.append(("minute:",minute))
    info.append(("second:",second))
    info.append(("millisecond:",millisecond))

    channel_names = []
    for channel_idx in range(0,number_channels):
        channel_names.append( (struct.unpack('4s', stream.read(4))[0]).decode('UTF-8') )
        #channel_names.append(struct.unpack('4s', stream.read(4))[0])
        stream.read(4)
    
    data = []
    dataTMP = struct.unpack(str(number_channels * number_timeframes) + "f", stream.read(number_channels * number_timeframes * 4))

    for idx in range(0,number_timeframes):
        data.append(dataTMP[idx*number_channels:(idx+1)*number_channels])
        
    infoArray = np.array(info)
    channelArray = np.array(channel_names)
    dataArray = np.array(data)
    
    return [infoArray,channelArray,dataArray]
# -------------------------------------------------------------------- #








def readSEFnew( filename ):
    stream = open(filename,'rb')
    version = struct.unpack('4s', stream.read(4))[0]
    number_channels = struct.unpack('i', stream.read(4))[0]
    number_aux_channels = struct.unpack('i', stream.read(4))[0]
    number_timeframes = struct.unpack('i', stream.read(4))[0]
    samplingrate = struct.unpack('f', stream.read(4))[0]
    year = struct.unpack('h', stream.read(2))[0]
    month = struct.unpack('h', stream.read(2))[0]
    day = struct.unpack('h', stream.read(2))[0]
    hour = struct.unpack('h', stream.read(2))[0]
    minute = struct.unpack('h', stream.read(2))[0]
    second = struct.unpack('h', stream.read(2))[0]
    millisecond = struct.unpack('h', stream.read(2))[0]
    
    info = []
    info.append(("version:",version))
    info.append(("number of channels:",number_channels))
    info.append(("number of aux channels:",number_aux_channels))
    info.append(("number of timeframes:",number_timeframes))
    info.append(("samplingrate:",samplingrate))
    info.append(("year:",year))
    info.append(("month:",month))
    info.append(("day:",day))
    info.append(("hour:",hour))
    info.append(("minute:",minute))
    info.append(("second:",second))
    info.append(("millisecond:",millisecond))

    channel_names = []
    for channel_idx in range(0,number_channels):
        channel_names.append( (struct.unpack('4s', stream.read(4))[0]).decode('UTF-8') )
        #channel_names.append(struct.unpack('4s', stream.read(4))[0])
        stream.read(4)
    
    data = []
    dataTMP = struct.unpack(str(number_channels * number_timeframes) + "f", stream.read(number_channels * number_timeframes * 4))

    for idx in range(0,number_timeframes):
        data.append(dataTMP[idx*number_channels:(idx+1)*number_channels])
        
    infoArray = np.array(info)
    channelArray = np.array(channel_names)
    dataArray = np.array(data)
    
    return [infoArray,channelArray,dataArray]
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def write( filename, infoArray, channelArray, dataArray):
    fout = open(filename, 'wb')
    # get number of channels and length of time series
    nTimeframes,nChannels = dataArray.shape

    # write: version
    string = infoArray[0,1]
    s = struct.Struct('4s')
    packed_data = s.pack(string.encode('utf-8'))
    fout.write( packed_data )
    
    # write: number of channels
    fout.write( struct.pack('i',int(nChannels)))
    
    # write: number of aux channels
    integer = int(infoArray[2,1])
    fout.write( struct.pack('i',integer))
    
    # write: number of timeframes
    fout.write( struct.pack('i',int(nTimeframes)))    
    
    # write: samplingrate
    floating = float(infoArray[4,1])
    fout.write( struct.pack('f',floating))      
    
    # write: year
    shortInt = int(infoArray[5,1])
    fout.write( struct.pack('h',shortInt))      

    # write: month
    shortInt = int(infoArray[6,1])
    fout.write( struct.pack('h',shortInt))    

    # write: day
    shortInt = int(infoArray[7,1])
    fout.write( struct.pack('h',shortInt))    

    # write: hour
    shortInt = int(infoArray[8,1])
    fout.write( struct.pack('h',shortInt))    

    # write: minute
    shortInt = int(infoArray[9,1])
    fout.write( struct.pack('h',shortInt))    

    # write: second
    shortInt = int(infoArray[10,1])
    fout.write( struct.pack('h',shortInt))    

    # write: millisecond
    shortInt = int(infoArray[11,1])
    fout.write( struct.pack('h',shortInt))

    # write: channels
    for channel_idx in range(0,nChannels):
        string = channelArray[channel_idx]
        s = struct.Struct('8s')
        packed_data = s.pack(string.encode('utf-8'))
        fout.write( packed_data )

    # write: data
    for idxTF in range(0,nTimeframes):
        for idxCh in range(0,nChannels):
            floating = float(dataArray[idxTF,idxCh])
            fout.write( struct.pack('f',floating))

    fout.close()
    
    return True
# -------------------------------------------------------------------- #
