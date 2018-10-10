import pandas as pd
from glob import glob
import pymysql
import sqlite3
import time

def read_file(file_dir,genre):
    x=glob(file_dir)
    thingstosave=[]

    for i in x:
        file=pd.read_json(i)
        for song in file['songs']:
            temp_dict = {}
            temp_dict['artist'] = song['artist']
            temp_dict['title'] = song['title']
            temp_dict['lyrics'] =song['lyrics']
            temp_dict['year'] = song['year']
            temp_dict['genre']=genre
            thingstosave += [temp_dict]
    data_list=[]
    for item in thingstosave:
        temp_list=[]
        temp_list.append(item['artist'])
        temp_list.append(item['title'])
        temp_list.append(item['lyrics'])
        temp_list.append(item['year'])
        temp_list.append(item['genre'])
        data_list.append(temp_list)
    return data_list

def save_in_mysql(data_list):
    conn=pymysql.connect(host='localhost',user='root',password='lbj23kg5',port=3306,charset='utf8mb4',db='lyricsdb')
    cur=conn.cursor()

    now=time.strftime('%M:%S')

    try:
        cur.executemany("insert into song values(%s,%s,%s,%s,%s)",data_list)
        conn.commit()
    except Exception as err:
        print(err)
    finally:
        cur.close()
        conn.close()
    end=time.strftime('%M:%S')

    return now+','+end

def save_in_sqlite(db_dir,data_list):
    conn=sqlite3.connect(db_dir)
    c=conn.cursor()
    c.executemany('insert into song value(?,?,?,?,?)',data_list)
    conn.commit()

def main():
    data=read_file('C:/Users/jackpork0702/Desktop/Lyrics/Rock_lyrics/*.json','Rock')
    print(len(data))
    save_in_mysql(data)

if __name__ == '__main__':
        main()