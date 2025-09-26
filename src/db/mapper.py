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


def update_train_status(dataset_id: int, status: str):
    sql = "update fingerprint_dataset set train_status = %s where id = %s"
    result = sql_update(sql, (status, dataset_id))
    if result:
        return True
    return False


def sql_select(sql: str, params: ()):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            selectSql = sql
            result = db.execute_query(selectSql, params)
            if result:
                return result


def sql_update(sql: str, params: ()):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            updateSql = sql
            db.execute_update(updateSql, params)
            return True
        return False
