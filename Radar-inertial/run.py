from utility.data_process import calc_all
from utility.drawing import draw
import multiprocessing as mp
import os


if __name__ == '__main__':
    index = os.listdir('./dataset2/0926')

    '''
    for i in range(len(index)):
        index[i] = '0926/' + index[i]
        calc_all(index[i])
    '''

    for i in range(len(index)):
        index[i] = '0926/' + index[i]
    pool = mp.Pool(12)
    pool.map(calc_all, index)
    pool.close()
    pool.join()
    