import configparser
import os

import mysql.connector
from mysql.connector import Error

from util.file_utils import find_root


class MySQLConnector:
    """MySQL数据库连接工具类"""

    def __init__(self, config_file=find_root() + '/config/database_config.ini'):
        """初始化，读取配置文件"""
        self.config = self._read_config(config_file)
        self.connection = None
        self.cursor = None

    def _read_config(self, config_file):
        """读取配置文件中的数据库连接信息"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")

        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')

        # 检查配置文件是否包含mysql部分
        if 'mysql' not in config.sections():
            raise ValueError("配置文件中未找到mysql配置部分")

        return {
            'host': config.get('mysql', 'host'),
            'port': config.getint('mysql', 'port'),
            'user': config.get('mysql', 'user'),
            'password': config.get('mysql', 'password'),
            'database': config.get('mysql', 'database'),
            'charset': config.get('mysql', 'charset', fallback='utf8mb4'),
            'connect_timeout': config.getint('mysql', 'connect_timeout', fallback=10)
        }

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                connect_timeout=self.config['connect_timeout']
            )

            if self.connection.is_connected():
                # 创建游标，设置返回结果为字典格式
                self.cursor = self.connection.cursor(dictionary=True)
                return True
            return False

        except Error as e:
            print(f"数据库连接失败: {str(e)}")
            self.connection = None
            self.cursor = None
            return False

    def execute_query(self, sql, params=None):
        """
        执行查询语句
        :param sql: SQL查询语句
        :param params: SQL参数，用于参数化查询
        :return: 查询结果列表
        """
        if not self.connection or not self.cursor or not self.connection.is_connected():
            # 如果连接已断开，尝试重新连接
            if not self.connect():
                return None

        try:
            self.cursor.execute(sql, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"查询执行失败: {str(e)}")
            return None

    def execute_update(self, sql, params=None):
        """
        执行更新语句(INSERT, UPDATE, DELETE等)
        :param sql: SQL语句
        :param params: SQL参数，用于参数化查询
        :return: 影响的行数，失败返回-1
        """
        if not self.connection or not self.cursor or not self.connection.is_connected():
            # 如果连接已断开，尝试重新连接
            if not self.connect():
                return -1

        try:
            self.cursor.execute(sql, params or ())
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            print(f"更新执行失败: {str(e)}")
            self.connection.rollback()
            return -1

    def execute_batch(self, sql, params_list):
        """
        批量执行更新语句
        :param sql: SQL语句
        :param params_list: 参数列表，每个元素是一个参数元组
        :return: 影响的行数总和，失败返回-1
        """
        if not self.connection or not self.cursor or not self.connection.is_connected():
            if not self.connect():
                return -1

        try:
            self.cursor.executemany(sql, params_list)
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            print(f"批量执行失败: {str(e)}")
            self.connection.rollback()
            return -1

    def get_last_insert_id(self):
        """获取最后插入记录的ID"""
        if self.cursor:
            return self.cursor.lastrowid
        return None

    def close(self):
        """关闭连接和游标"""
        if self.cursor:
            try:
                self.cursor.close()
            except Error as e:
                print(f"关闭游标失败: {str(e)}")

        if self.connection and self.connection.is_connected():
            try:
                self.connection.close()
            except Error as e:
                print(f"关闭连接失败: {str(e)}")

        self.cursor = None
        self.connection = None

    def __enter__(self):
        """支持上下文管理器，with语句"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时关闭连接"""
        self.close()

# 使用示例
