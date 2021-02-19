import csv
import time
import datetime
from makeprediction.invtools import date2num


def rtts(function, step = 1, filename = None, fieldnames = None):
    
    if fieldnames is None:
        fieldnames = ["date", "value"]

    if filename is None:
        filename = 'data_ts.csv'

    with open(filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()


    try:
        while True:

            with open(filename, 'a') as csv_file:
                x = datetime.datetime.now()
                x = x.replace(microsecond = 0) + datetime.timedelta(seconds = step)

                dt = date2num(x)
                #line = dt - date2num(x.strftime('%Y-%m-%d'))
                
                y = function(dt)

                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                info = {
                    fieldnames[0]:  x.strftime("%m/%d/%Y, %H:%M:%S"),
                    fieldnames[1]: y,
                }
                t = datetime.datetime.now()
                
                #if t == x + step:
                csv_writer.writerow(info)
                    #x = t


                time.sleep((x-t).total_seconds())
                print(info)
                #print((x-t).total_seconds())

    except KeyboardInterrupt:
        print('Data generation interrupted by user.')

            