import psycopg2
try:
    conn = psycopg2.connect(user='postgres',
                            database='postgres',
                            password='postgres', 
                            host='172.20.0.3')
    conn.close()
    print("Success")
except psycopg2.OperationalError as ex:
    print("Connection failed: {0}".format(ex))