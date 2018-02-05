
from sqlalchemy import create_engine
import MySQLdb
from pymongo import MongoClient
import pandas as pd


class Mysql_helper(object):

    def __init__(self, database='option'):
        self.engine = create_engine('mysql://root:@localhost/{}'.format(database))
        self.db = MySQLdb.connect('localhost', 'root', '', database)
        self.cur = self.db.cursor()

    def insert(self, data, table_name, if_exists='append'):
        data.to_sql(table_name, self.engine, if_exists=if_exists, index=False, chunksize=1000)

    def read(self, table_name):
        data = pd.read_sql(table_name, self.engine)
        return data

    def query(self, sql_query):
        data = pd.read_sql_query(sql_query, self.engine)
        return data

    def sql(self, sql):
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            # Rollback in case there is any error
            self.db.rollback()


class Mongo_helper():
    def __init__(self, database='option', url='localhost:27017', table=None):
        self.client = MongoClient('mongodb://{}'.format(url))
        self.database = self.client[database]
        if table is not None:
            self.table = self.client[database][table]
        else:
            self.table = self.database[self.database.collection_names()[0]]

    def insert(self, data):
        if type(data) == pd.core.frame.DataFrame:
            self.table.insert_many(data.to_dict('records'))
        else:
            self.table.insert_many(data)

    def read_df(self, ):
        df = pd.DataFrame(list(self.table.find()))
        return df.drop('_id', axis=1) if df.shape[0] != 0 else None

    def query(self, conditions):
        df = pd.DataFrame(list(self.table.find(eval(conditions))))
        return df.drop('_id', axis=1) if df.shape[0] != 0 else None

    def update(self, conditions, updata):
        self.table.update_many(eval(conditions), {"$set": eval(updata)})

    def delete(self, conditions):
        self.table.delete_many(eval(conditions))

