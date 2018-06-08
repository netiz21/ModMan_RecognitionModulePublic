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

def capture_and_send_recv(soc, USE_FAKE_CAPTURE=False, cmd='r'):
    if USE_FAKE_CAPTURE == False:
        ret, frame = cap.read()
    else:
        ret = True
        # frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.int8)
        frame = cv2.imread('./iPadShot.png', cv2.IMREAD_COLOR)

    if ret is True:
        print('Client: image captured')

        '''
        Send
        '''
        msg = 'MMS'

        # cmd
        msg = msg + struct.pack('c', 'e')
        msg = msg + struct.pack('c', cmd)

        if cmd == 'c' or cmd == 'r':
            # intrisic
            msg = msg + struct.pack('f', 2400.970)  # fx
            msg = msg + struct.pack('f', 2398.881)  # fy
            msg = msg + struct.pack('f', 1096.002)  # px
            msg = msg + struct.pack('f', 815.355)   # py

        # RMat, tVec
        for j in range(0, 12):
            msg = msg + struct.pack('f', float(j))  # 36 bytes

        if cmd == 'c' or cmd == 'r':
            # # Image
            # for i in range(IMG_HEIGHT):
            #     for j in range(IMG_WIDTH):
            #         for c in range(3):
            #             str_c = '%c'%frame[i, j, c]
            #             # msg = msg + struct.pack('c', str_c.encode('ascii'))
            #             msg = msg + struct.pack('c', str_c)

            # Image
            for i in range(IMG_HEIGHT/4):
                for j in range(IMG_WIDTH):
                    for c in range(3):
                        str_c = '%c' % frame[i, j, c]
                        # msg = msg + struct.pack('c', str_c.encode('ascii'))
                        msg = msg + struct.pack('c', str_c)

            print('Client: try to send.')
            soc.send(msg)
            print('Client: msg sent: len %d' % len(msg))

            msg = ''
            for i in range(IMG_HEIGHT/4, IMG_HEIGHT/2):
                for j in range(IMG_WIDTH):
                    for c in range(3):
                        str_c = '%c' % frame[i, j, c]
                        # msg = msg + struct.pack('c', str_c.encode('ascii'))
                        msg = msg + struct.pack('c', str_c)

            print('Client: try to send.')
            soc.send(msg)
            print('Client: msg sent: len %d' % len(msg))

            msg = ''
            for i in range(IMG_HEIGHT / 2, 3*IMG_HEIGHT / 4):
                for j in range(IMG_WIDTH):
                    for c in range(3):
                        str_c = '%c' % frame[i, j, c]
                        # msg = msg + struct.pack('c', str_c.encode('ascii'))
                        msg = msg + struct.pack('c', str_c)

            print('Client: try to send.')
            soc.send(msg)
            print('Client: msg sent: len %d' % len(msg))

            msg = ''
            for i in range(3*IMG_HEIGHT / 4, IMG_HEIGHT):
                for j in range(IMG_WIDTH):
                    for c in range(3):
                        str_c = '%c' % frame[i, j, c]
                        # msg = msg + struct.pack('c', str_c.encode('ascii'))
                        msg = msg + struct.pack('c', str_c)

        msg = msg + 'MME'

        try:
            print('Client: try to send.')
            soc.send(msg)
            print('Client: msg sent: len %d'%len(msg))
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
        if(cmd == 'r'):
            print('Client: try to recv.')
            data = soc.recv(4096)

            if 'MMS' in data:
                print('receive data')

                index_start = data.index('MMS')
                index_end = data.index('MME')

                char_inst = struct.unpack('c', data[index_start+3:index_start+4])[0]
                char_cmd  = struct.unpack('c', data[index_start+4:index_start+5])[0]
                print('%c%c'%(char_inst, char_cmd))

                num_objs = struct.unpack('i', data[index_start+5:index_start+9])[0]
                print('n_objs: %d\n'%num_objs)

                len_one_obj = 1 + 4*12 + 4*4
                for iObj in range(num_objs):
                    data_one_obj = data[index_start+9+iObj*len_one_obj : index_start+9+(iObj+1)*len_one_obj]

                    value = struct.unpack('c', data_one_obj[0])[0] # 1byte
                    print('C: %c'%value)

                    print('Rot: \n')
                    for j in range(0, 9):
                        value = struct.unpack('f', data_one_obj[1+j*4 : 1+(j+1)*4])[0]   # 36 bytes
                        print('%f '%value)

                    print('Trn: \n')
                    for j in range(0, 3):
                        value = struct.unpack('f', data_one_obj[37+j*4 : 37+(j+1)*4])[0]  # 12 bytes
                        print('%f '%value)

                    print('left, top, right, bottom: \n')   # 4*4 bytes
                    value = struct.unpack('i', data_one_obj[49:53])[0]
                    print('%d '%value)
                    value = struct.unpack('i', data_one_obj[53:57])[0]
                    print('%d ' % value)
                    value = struct.unpack('i', data_one_obj[57:61])[0]
                    print('%d ' % value)
                    value = struct.unpack('i', data_one_obj[61:65])[0]
                    print('%d ' % value)

    print('Client: done')

    return True

if __name__ == '__main__':
    USE_FAKE_CAPTURE = True

    '''
    data info
    '''
    IMG_WIDTH = 2224
    IMG_HEIGHT = 1668

    if not USE_FAKE_CAPTURE:
        cap = cv2.VideoCapture(0)
        cap.set(3, IMG_WIDTH)
        cap.set(4, IMG_HEIGHT)

    '''
    network info
    '''
    IP = '129.254.87.77' #'127.0.0.1', '192.168.137.50'
    PORT= 8020
    ADDR = (IP, PORT)
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        print('Client: trying to connect the server')
        soc.connect(ADDR)
    except Exception as e:
        print('Client: cannot connect the server')
        sys.exit()
    print('Client: connected to the server')


    while True:
        cmd = input('Choose command (ex: \'c\'alibrate or \'r\'ecognize or \'d\'etect).\n>>')

        if cmd == 'c' or cmd == 'r' or cmd == 'd':
            capture_and_send_recv(soc, USE_FAKE_CAPTURE=USE_FAKE_CAPTURE, cmd = cmd)
        else:
            pass
