# coding=utf-8
# #!/usr/bin/env python

# network couterpart simulation
# working as client or server

import cv2
import numpy as np

# socket
import os, sys
import socket, errno
import struct

def capture_and_send_recv(soc, USE_FAKE_CAPTURE=False):
    if USE_FAKE_CAPTURE == False:
        ret, frame = cap.read()
    else:
        ret = True
        frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.int8)

    if ret is True:
        print('Client: image captured')

        '''
        Send
        '''
        msg = 'MMS'

        for i in range(IMG_HEIGHT):
            for j in range(IMG_WIDTH):
                for c in range(3):
                    str_c = '%c'%frame[i, j, c]
                    # msg = msg + struct.pack('c', str_c.encode('ascii'))
                    msg = msg + struct.pack('c', str_c)

        msg = msg + 'MME'

        try:
            print('Client: try to send.')
            soc.send(msg)
        except socket.error, e:
            if isinstance(e.args, tuple):
                print "errno is %d" % e[0]
                if e[0] == errno.EPIPE:
                    # remote peer disconnected
                    print "Detected remote disconnect"
                else:
                    # determine and handle different error
                    pass
            else:
                print "socket error ", e
            soc.close()
            return False
        '''
        Receive
        '''
        print('Client: try to recv.')
        data = soc.recv(4096)

        if 'MMS' in data:
            print('receive data')

            index_start = data.index('MMS')
            index_end = data.index('MME')

            num_objs = struct.unpack('i', data[index_start+3:index_start+7])[0]

            len_one_obj = 1 + 48 + 16   # c + float x (9 + 3) + int x 4
            for iObj in range(num_objs):
                data_one_obj = data[index_start+7+iObj*len_one_obj : index_start+7+(iObj+1)*len_one_obj]

                value = struct.unpack('c', data_one_obj[0])[0] # 1byte
                print('C: %c'%value)

                print('Rot: \n')
                for j in range(0, 9):
                    value = struct.unpack('f', data_one_obj[1+j*4 : 1+(j+1)*4])   # 36 bytes
                    print('%f '%value)

                print('Trn: \n')
                for j in range(0, 3):
                    value = struct.unpack('f', data_one_obj[37+j*4 : 37+(j+1)*4])  # 12 bytes
                    print('%f '%value)

                print('left, top, right, bottom: \n')
                value = struct.unpack('i', data_one_obj[49:53])
                print('%d '%value)
                value = struct.unpack('i', data_one_obj[53:57])
                print('%d ' % value)
                value = struct.unpack('i', data_one_obj[57:61])
                print('%d ' % value)
                value = struct.unpack('i', data_one_obj[61:65])
                print('%d ' % value)

    return True

if __name__ == '__main__':
    DO_AS_SERVER = False # False == as client
    USE_FAKE_CAPTURE = False

    '''
    data info
    '''
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    NET_BUFSIZE = (IMG_WIDTH * IMG_HEIGHT * 3 + 6)

    if not USE_FAKE_CAPTURE:
        cap = cv2.VideoCapture(0)
        cap.set(3, IMG_WIDTH)
        cap.set(4, IMG_HEIGHT)

    '''
    network info
    '''
    IP = '129.254.87.77' #'127.0.0.1'
    PORT= 8020
    ADDR = (IP, PORT)
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if DO_AS_SERVER == False:
        try:
            print('Client: trying to connect the server')
            soc.connect(ADDR)
        except Exception as e:
            print('Client: cannot connect the server')
            sys.exit()
        print('Client: connected to the server')


        while True:
            capture_and_send_recv(soc, USE_FAKE_CAPTURE=USE_FAKE_CAPTURE)

    else:
        soc.bind(ADDR)

        while True:
            print('Server: waiting of client connection')
            soc.listen(1)   # max num of connection for req.
            clientSocket, clientAddr = soc.accept()
            print('Server: connected to the client (%s:%s)' % clientAddr)

            while True:
                ret = capture_and_send_recv(clientSocket)

                if ret == False:
                    break




