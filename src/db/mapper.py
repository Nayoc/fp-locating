from util.mysql_utils import MySQLConnector


def select_space(space_id: int):
    sql = "select * from indoor_space where id = %s limit 1"
    result = sql_select(sql, (space_id,))
    if result:
        return result[0]
    return None


def select_fingerprint_dataset(dataset_id: int):
    sql = "select * from fingerprint_dataset where id = %s order by create_time desc limit 1"
    result = sql_select(sql, (dataset_id,))
    if result:
        return result[0]
    return None


def update_dataset_status(dataset_id: int, status: str):
    sql = "update fingerprint_dataset set train_status = %s where id = %s"
    result = sql_update(sql, (status, dataset_id))
    if result:
        return True
    return False


def update_dataset_model(dataset_id: int, model_file: str):
    sql = "update fingerprint_dataset set model_file = %s where id = %s"
    result = sql_update(sql, (model_file, dataset_id))
    if result:
        return True
    return False


def sql_select(sql: str, params: ()):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            select_sql = sql
            result = db.execute_query(select_sql, params)
            if result:
                return result


def sql_update(sql: str, params: ()):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            update_sql = sql
            db.execute_update(update_sql, params)
            return True
        return False


def sql_insert(sql: str, params: ()):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            insert_sql = sql
            db.execute_insert(insert_sql, params)
            return True
        return False
