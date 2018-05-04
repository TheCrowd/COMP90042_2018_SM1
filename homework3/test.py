# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:09:37 2018

@author: zsl
"""


import json,time,math,operator
from mpi4py import MPI
import numpy as np

filename = "tinyInstagram.json"
gridfile = "melbGrid.json"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

length_of_file = 0
buffer_length = 0
chunk_size = 8
#grid = {}
count = 0
list = []
ins = {}
insdata = []
result = {'A1':0,'A2':0,'A3':0,'A4':0,
          'B1':0,'B2':0,'B3':0,'B4':0,
          'C1':0,'C2':0,'C3':0,'C4':0,'C5':0,
          'D3':0,'D4':0,'D5':0
          }
result_pro = np.zeros(16)
comm.Barrier()
start_time = time.time()

with open(gridfile, encoding='utf-8', mode = 'r') as g:
    grid = json.load(g)

Aymin = grid['features'][0]['properties']['ymin']
Aymax = grid['features'][0]['properties']['ymax']
Bymin = grid['features'][4]['properties']['ymin']
Bymax = grid['features'][4]['properties']['ymax']
Cymin = grid['features'][8]['properties']['ymin']
Cymax = grid['features'][8]['properties']['ymax']
Dymin = grid['features'][13]['properties']['ymin']
Dymax = grid['features'][13]['properties']['ymax']

Col1 = grid['features'][0]['properties']['xmin']
Col2 = grid['features'][1]['properties']['xmin']
Col3 = grid['features'][2]['properties']['xmin']
Col4 = grid['features'][3]['properties']['xmin']
Col5 = grid['features'][15]['properties']['xmin']
Col6 = grid['features'][15]['properties']['xmax']

with open(filename, encoding='utf-8', mode = 'r') as f:
    for line in f:
        try:
            #insdata.append(json.loads(line[:-2]))
            length_of_file += 1
        except:
            pass
#print(length_of_file)
if size > 1:
    length = int(math.ceil(length_of_file/size)) 
    #buffer = np.empty(buffer_length)
    offset = length * rank
#
    count = 0
    insdata = []
    with open(filename, encoding='utf-8', mode = 'r') as f:
        for line in f:
            try:
                #insdata.append(json.loads(line[:-2]))
                count += 1
                if count > offset and count < offset + length:
                    insdata.append(json.load(line[:-2]))

            except:
                pass
    #buffer = insdata[offset:offset + buffer_length]
    
        

    try:
        for item in insdata:
            for i in item.keys():
                if i == 'doc':
                    for j in item[i].keys():
                        if j == 'coordinates':
                            for m in item[i][j].keys():
                                if m == 'coordinates':
                                    if (item[i][j][m][0] > Aymin
                                    and item[i][j][m][0] < Aymax):
                                        ##A
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[0] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[1] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[2] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[3] += 1
                                    if (item[i][j][m][0] > Bymin
                                    and item[i][j][m][0] < Bymax):
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[4] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[5] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[6] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[7] += 1
                                                    
                                    if (item[i][j][m][0] > Cymin
                                    and item[i][j][m][0] < Cymax):
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[8] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[9] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[10] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[11] += 1
                                            elif item[i][j][m][1] < Col6:
                                                result_pro[12] += 1
                                                
                                    if (item[i][j][m][0] > Dymin
                                    and item[i][j][m][0] < Dymax):
                                        if item[i][j][m][1] > Col3:
                                            if item[i][j][m][1] < Col4:
                                                result_pro[13] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[14] += 1
                                            elif item[i][j][m][1] < Col6:
                                                result_pro[15] += 1
    except:
        pass
    print ('[%i]'%comm.rank, result_pro)
    comm.Barrier()
    
else:
    length = int(math.ceil(length_of_file/chunk_size)) 
    #buffer = np.empty(buffer_length)
    for i in range(chunk_size):
        offset = length * i
#
        count = 0
        insdata = []
        with open(filename, encoding='utf-8', mode = 'r') as f:
            for line in f:
                try:
                    #insdata.append(json.loads(line[:-2]))
                    count += 1
                    #print(offset)
                    if (count > offset and count < offset + length):
                        insdata.append(json.load(line[:-2]))
                        print('x')
                
                except:
                    pass
        #buffer = insdata[offset:offset + buffer_length]
#
    print(count,insdata)
    print(offset)
    
    try:
        for item in insdata:
            for i in item.keys():
                if i == 'doc':
                    for j in item[i].keys():
                        if j == 'coordinates':
                            for m in item[i][j].keys():
                                if m == 'coordinates':
                                    if (item[i][j][m][0] > Aymin
                                    and item[i][j][m][0] < Aymax):
                                        ##A
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[0] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[1] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[2] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[3] += 1
                                    if (item[i][j][m][0] > Bymin
                                    and item[i][j][m][0] < Bymax):
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[4] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[5] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[6] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[7] += 1
                                                    
                                    if (item[i][j][m][0] > Cymin
                                    and item[i][j][m][0] < Cymax):
                                        if item[i][j][m][1] > Col1:
                                            if item[i][j][m][1] < Col2:
                                                result_pro[8] += 1
                                            elif item[i][j][m][1] < Col3:
                                                result_pro[9] += 1
                                            elif item[i][j][m][1] < Col4:
                                                result_pro[10] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[11] += 1
                                            elif item[i][j][m][1] < Col6:
                                                result_pro[12] += 1
                                                
                                    if (item[i][j][m][0] > Dymin
                                    and item[i][j][m][0] < Dymax):
                                        if item[i][j][m][1] > Col3:
                                            if item[i][j][m][1] < Col4:
                                                result_pro[13] += 1
                                            elif item[i][j][m][1] < Col5:
                                                result_pro[14] += 1
                                            elif item[i][j][m][1] < Col6:
                                                result_pro[15] += 1
    except:
        pass
    print ('[%i]'%comm.rank, result_pro)
    comm.Barrier()

if rank==0:
    # only processor 0 will actually get the data
    result1 = np.zeros_like(result_pro)
else:
    result1 = None

comm.Reduce(
    [result_pro, MPI.DOUBLE],
    [result1, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
    )
#print ('[%i]'%comm.rank, result1)

comm.Barrier()

t_diff = time.time() - start_time
##
if rank==0:
    t_diff = time.time() - start_time
    print('Running time:' , t_diff)
if rank == 0:
    result['A1'] = result1[0]
    result['A2'] = result1[1]
    result['A3'] = result1[2]
    result['A4'] = result1[3]
    result['B1'] = result1[4]
    result['B2'] = result1[5]
    result['B3'] = result1[6]
    result['B4'] = result1[7]
    result['C1'] = result1[8]
    result['C2'] = result1[9]
    result['C3'] = result1[10]
    result['C4'] = result1[11]
    result['C5'] = result1[12]
    result['D3'] = result1[13]
    result['D4'] = result1[14]
    result['D5'] = result1[15]
    sorted_x = sorted(result.items(), key=operator.itemgetter(1),reverse = True)
    print('Order the Grid boxes based on the total number of posts made in each box')
    for i in range(len(sorted_x)):
        print(sorted_x[i][0], ' :',sorted_x[i][1], ' posts' )
    print('-----------------------------------------------------------------')
    print('Order the rows based on the total number of posts in each row')
    result_row = {'A-Row':0, 'B-Row':0, 'C-Row':0, 'D-Row':0}
    result_row['A-Row'] = result['A1'] + result['A2'] + result['A3'] + result['A4']
    result_row['B-Row'] = result['B1'] + result['B2'] + result['B3'] + result['B4']
    result_row['C-Row'] = result['C1'] + result['C2'] + result['C3'] + result['C4'] + result['C5']
    result_row['D-Row'] = result['D3'] + result['D4'] + result['D5']
    sorted_row = sorted(result_row.items(), key=operator.itemgetter(1),reverse = True)
    for i in range(len(result_row)):
        print(sorted_row[i][0], ' :',sorted_row[i][1], ' posts' )
    print('-----------------------------------------------------------------')
    print('Order the columns based on the total number of posts in each column')
    result_col = {'Column 1':0, 'Column 2':0, 'Column 3':0, 'Column 4':0, 'Column 5':0}
    result_col['Column 1'] = result['A1'] + result['B1'] + result['C1']
    result_col['Column 2'] = result['A2'] + result['B2'] + result['C2']
    result_col['Column 3'] = result['A3'] + result['B3'] + result['C3'] + result['D3']
    result_col['Column 4'] = result['A4'] + result['B4'] + result['C4'] + result['D4']
    result_col['Column 5'] = result['C5'] + result['D5']
    sorted_col = sorted(result_col.items(), key=operator.itemgetter(1),reverse = True)
    for i in range(len(sorted_col)):
        print(sorted_col[i][0], ' :',sorted_col[i][1], ' posts' )

