# -*- coding: utf-8 -*-
'''
ischwaninger
19 April 2018

load wickedonna tweets and random weibo tweets from sqlite database
'''


import sqlite3

#get clean_text, label fom sqlite database table
#@random = true for tweets in random order
def load_data_from_sqlite(db, table, max_len, min_len, limit, random=True):

    #connect to db
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    #fetch data
    if random:
        statement="SELECT clean_text, label FROM %s LIMIT 500000;"% (table)
    else:
        statement="SELECT clean_text, label FROM %s LIMIT 85000;"% (table)
    cur.execute(statement)
    data = cur.fetchall()

    X = []
    y = [] 

    count = 0
    for entry in data:
        if len(entry[0]) <= max_len and len(entry[0]) >= min_len:
            X.append(entry[0])
            y.append(entry[1])
            count += 1
            if count == limit:
                return X, y
    return X, y

#get clean_text, label fom sqlite database table - testing database
#@random = true for tweets in random order
def load_data_from_sqlite_test(db, table, max_len, min_len, limit, random=True):

    #connect to db
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    #fetch data
    if random:
        statement="SELECT clean_text, label FROM %s WHERE label NOT LIKE 'random' ORDER BY random() LIMIT 2000000;"% (table)
    else:
        statement="SELECT clean_text, label FROM %s WHERE label NOT LIKE 'random' LIMIT 2000000;"% (table)
    cur.execute(statement)
    data = cur.fetchall()

    X = []
    y = [] 

    count = 0
    for entry in data:
        if len(entry[0]) <= max_len and len(entry[0]) >= min_len:
            X.append(entry[0])
            y.append(entry[1].split('_')[0])
            count += 1
            if count == limit:
                return X, y
    return X, y

def test_load_data_from_sqlite(db, table, max_len, min_len, limit, random):
    X, y = load_from_sqlite(db, table, max_len, min_len, limit, random)
    print(X[0], y[0])
    assert len(X) == len(y)
    print(len(X))

if __name__ == '__main__':
    test_load_data_from_sqlite("Tweets_Protest_Random.db", "ProtestTweets", 280, 3, 75000, random="true")
