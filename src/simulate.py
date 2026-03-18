import random
import time
from util.mysql_utils import MySQLConnector

wifi_str = """
9a:4a:6b:57:18:fe,
9a:4a:6b:93:0b:7a,
de:a7:82:43:5e:00,
c2:a4:76:98:1b:2c,
9a:4a:6b:82:be:82,
9a:4a:6b:97:18:fe,
9a:4a:6b:17:18:fe,
9a:4a:6b:92:be:82,
c2:a4:76:88:1b:2c,
9a:4a:6b:83:0b:7a,
"""

max_num = 100

demo_fill_map = {
    "11_10.25": {
        "cell": [
            {
                "ap_id": '268',
                "target_rsrp": -80,
                "target_rsrq": -11,
                "target_sinr": 11,
                "request_batch_id": [1, 2, 3]
            },
            {
                "ap_id": '17',
                "target_rsrp": -100,
                "target_rsrq": -11,
                "target_sinr": 11,
                "request_batch_id": [2, 3, 4]
            },
        ],
        "wifi": [
            {
                "ap_id": 'w1',
                "ap_name": 'w1n',
                "target_rssi": -80,
                "request_batch_id": [1, 2, 3, 5, 7]
            },
            {
                "ap_id": 'w2',
                "ap_name": 'w2n',
                "target_rssi": -80,
                "request_batch_id": [1, 2, 3, 5, 6, 7, 8]
            },
        ],
    }
}

collections_batch_id = 1772167533230

def build_fill_map():
    # 初始化最终的fill_map结构
    fill_map = {}

    with MySQLConnector() as db:
        if db.connection.is_connected():
            # 1. 执行Cell和WiFi的分组查询
            cell_group_sql = """
                select rp_x,rp_y,ap_id,
                       round(avg(ap_rsrp)) target_rsrp,
                       round(avg(ap_rsrq)) target_rsrq,
                       round(avg(ap_sinr)) target_sinr,
                       group_concat(request_batch_id) as request_batch 
                from single_collection_data 
                where source='cell' and space_id=19 
                group by rp_x,rp_y,ap_id 
                order by ap_id asc,rp_x desc,rp_y desc
            """  # 注：原SQL的group by多了ap_name，cell的source下可能无ap_name，已修正
            wifi_group_sql = """
                select rp_x,rp_y,ap_id,ap_name,
                       round(avg(ap_rssi)) target_rssi,
                       group_concat(request_batch_id) as request_batch 
                from single_collection_data 
                where source='wifi' and space_id=19 
                  and ap_id in ('58:41:20:39:4e:0b',
                                'a4:a9:30:c6:88:6f',
                                'a4:a9:30:ec:cf:63',
                                '64:6e:97:e7:c0:72',
                                'a4:a9:30:ec:cf:62',
                                'a4:a9:30:c6:88:6e',
                                '24:5a:5f:81:bc:40',
                                '4c:c6:4c:b1:a9:66',
                                'a4:a9:30:c6:85:2e',
                                '16:cb:19:99:2f:b1') 
                group by rp_x,rp_y,ap_id,ap_name 
                order by ap_name asc,rp_x desc,rp_y desc
            """
            cell_results = db.execute_query(cell_group_sql)
            wifi_results = db.execute_query(wifi_group_sql)

            # 2. 处理Cell查询结果，填充到fill_map
            for row in cell_results:
                # 提取基础字段（处理NULL值，避免报错）
                rp_x = row.get('rp_x', '') or ''
                rp_y = row.get('rp_y', '') or ''
                ap_id = row.get('ap_id', '') or ''
                target_rsrp = row.get('target_rsrp', 0) or 0
                target_rsrq = row.get('target_rsrq', 0) or 0
                target_sinr = row.get('target_sinr', 0) or 0
                request_batch_str = row.get('request_batch', '') or ''

                # 拼接坐标键（和之前的fill_map格式一致：rp_x_rp_y）
                coord_key = f"{rp_x}_{rp_y}"

                # 转换request_batch：字符串转整数列表（处理空值/分隔符）
                request_batch_id = []
                if request_batch_str.strip():
                    request_batch_id = [int(batch.strip()) for batch in request_batch_str.split(',') if batch.strip()]

                # 初始化coord_key对应的结构（避免KeyError）
                if coord_key not in fill_map:
                    fill_map[coord_key] = {
                        "cell": [],
                        "wifi": []
                    }

                # 填充Cell AP数据到对应坐标
                fill_map[coord_key]['cell'].append({
                    "ap_id": ap_id,
                    "target_rsrp": target_rsrp,
                    "target_rsrq": target_rsrq,
                    "target_sinr": target_sinr,
                    "request_batch_id": request_batch_id
                })

            # 3. 处理WiFi查询结果，填充到fill_map
            for row in wifi_results:
                # 提取基础字段（处理NULL值）
                rp_x = row.get('rp_x', '') or ''
                rp_y = row.get('rp_y', '') or ''
                ap_id = row.get('ap_id', '') or ''
                ap_name = row.get('ap_name', '') or ''
                target_rssi = row.get('target_rssi', 0) or 0
                request_batch_str = row.get('request_batch', '') or ''

                # 拼接坐标键
                coord_key = f"{rp_x}_{rp_y}"

                # 转换request_batch为整数列表
                request_batch_id = []
                if request_batch_str.strip():
                    request_batch_id = [int(batch.strip()) for batch in request_batch_str.split(',') if batch.strip()]

                # 初始化coord_key对应的结构
                if coord_key not in fill_map:
                    fill_map[coord_key] = {
                        "cell": [],
                        "wifi": []
                    }

                # 填充WiFi AP数据到对应坐标
                fill_map[coord_key]['wifi'].append({
                    "ap_id": ap_id,
                    "ap_name": ap_name,
                    "target_rssi": target_rssi,
                    "request_batch_id": request_batch_id
                })

    # 返回最终构建的fill_map
    return fill_map


def build_cell_sql(fill_map: dict):
    # 修正WiFi的source字段（原错误写为'cell'）
    CELL_SQL_INSERT = "insert into single_collection_data (space_id, collection_batch_id, request_batch_id, ap_id, ap_rsrp, ap_rsrq, ap_sinr, rp_x, rp_y, source,type) values\n"
    CELL_SQL_VALUE = "(19,1772620014321,{request_batch_id},'{ap_id}',{ap_rsrp},{ap_rsrq},{ap_sinr},{rp_x},{rp_y},'cell',2),\n"

    WIFI_SQL_INSERT = "insert into single_collection_data (space_id, collection_batch_id, request_batch_id, ap_id,ap_name, ap_rssi, rp_x, rp_y, source,type) values\n"
    WIFI_SQL_VALUE = "(19,1772620014321,{request_batch_id},'{ap_id}','{ap_name}',{ap_rssi},{rp_x},{rp_y},'wifi',2),\n"

    sql_output_file = f"fill_all_sql_{int(time.time() * 1000)}.sql"
    # 初始化文件（清空原有内容，保证每次运行重新生成）
    with open(sql_output_file, 'w', encoding='utf-8') as f:
        f.write(f"-- 信号补全SQL文件，生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for coord, apx in fill_map.items():
        coord_array = coord.split('_')
        coord_x = coord_array[0]
        coord_y = coord_array[1]

        cell_aps = apx['cell']
        wifi_aps = apx['wifi']

        # 计算坐标点公共批次并集
        cell_batch_ids = set([batch_id for ap in cell_aps for batch_id in ap['request_batch_id']])
        wifi_batch_ids = set([batch_id for ap in wifi_aps for batch_id in ap['request_batch_id']])
        global_batch_ids = cell_batch_ids | wifi_batch_ids
        global_batch_sorted = sorted(global_batch_ids)
        global_batch_len = len(global_batch_ids)

        # 生成坐标点级别的统一新批次列表（所有AP共用）
        new_batch_count = max(0, max_num - global_batch_len)
        new_batch_list = []
        if new_batch_count > 0:
            base_timestamp = int(time.time() * 1000)
            # 一行式生成随机8~12秒间隔的新批次
            new_batch_list = [base_timestamp + sum(random.randint(8000, 12000) for _ in range(i)) for i in
                              range(new_batch_count)]

        # ===================== 核心修正：移出缩进，保证始终处理Cell/WiFi =====================
        # ===================== 处理Cell AP：每个AP独立补全至max_num =====================
        cell_final_sql = ""
        for ap in cell_aps:
            ap_id = ap['ap_id']
            base_rsrp = ap['target_rsrp']
            base_rsrq = ap['target_rsrq']
            base_sinr = ap['target_sinr']
            own_batch = set(ap['request_batch_id'])

            ap_cell_sql = CELL_SQL_INSERT
            record_count = 0

            # 步骤1：补公共批次中自身缺失的部分
            fill_batch = sorted(global_batch_ids - own_batch)
            for bid in fill_batch:
                if record_count >= max_num:
                    break
                # 随机波动
                rsrp_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
                rsrq_offset = random.choice([-1, 0, 0, 0, 1])
                sinr_offset = random.choice([-1, 0, 0, 0, 1])
                target_rsrp = base_rsrp + rsrp_offset
                target_rsrq = base_rsrq + rsrq_offset
                target_sinr = base_sinr + sinr_offset

                ap_cell_sql += CELL_SQL_VALUE.format(
                    collection_batch_id=collections_batch_id,
                    request_batch_id=bid,
                    ap_id=ap_id,
                    ap_rsrp=target_rsrp,
                    ap_rsrq=target_rsrq,
                    ap_sinr=target_sinr,
                    rp_x=coord_x,
                    rp_y=coord_y
                )
                record_count += 1

            # 步骤2：补新批次（即使new_batch_count=0，也不影响循环逻辑）
            if record_count < max_num and new_batch_count > 0:
                need_new = max_num - record_count
                use_new_batches = new_batch_list[:need_new]
                for new_bid in use_new_batches:
                    rsrp_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
                    rsrq_offset = random.choice([-1, 0, 0, 0, 1])
                    sinr_offset = random.choice([-1, 0, 0, 0, 1])
                    target_rsrp = base_rsrp + rsrp_offset
                    target_rsrq = base_rsrq + rsrq_offset
                    target_sinr = base_sinr + sinr_offset

                    ap_cell_sql += CELL_SQL_VALUE.format(
                        collection_batch_id=collections_batch_id,
                        request_batch_id=new_bid,
                        ap_id=ap_id,
                        ap_rsrp=target_rsrp,
                        ap_rsrq=target_rsrq,
                        ap_sinr=target_sinr,
                        rp_x=coord_x,
                        rp_y=coord_y
                    )
                    record_count += 1

            # 处理SQL结尾
            ap_cell_sql = ap_cell_sql.rstrip(',\n') + ';'

            # ========== 关键修改1：判断是否为空SQL语句 ==========
            # 检查是否只有insert开头+values;，没有实际数据
            if ap_cell_sql.strip() == CELL_SQL_INSERT.rstrip('\n') + ';':
                continue  # 跳过空SQL
            cell_final_sql += ap_cell_sql + "\n\n"

        # ===================== 处理WiFi AP：每个AP独立补全至max_num =====================
        wifi_final_sql = ""
        for ap in wifi_aps:
            ap_id = ap['ap_id']
            ap_name = ap['ap_name']
            base_rssi = ap['target_rssi']
            own_batch = set(ap['request_batch_id'])

            ap_wifi_sql = WIFI_SQL_INSERT
            record_count = 0

            # 步骤1：补公共批次中自身缺失的部分
            fill_batch = sorted(global_batch_ids - own_batch)
            for bid in fill_batch:
                if record_count >= max_num:
                    break
                rssi_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
                target_rssi = base_rssi + rssi_offset

                ap_wifi_sql += WIFI_SQL_VALUE.format(
                    collection_batch_id=collections_batch_id,
                    request_batch_id=bid,
                    ap_id=ap_id,
                    ap_name=ap_name,
                    ap_rssi=target_rssi,
                    rp_x=coord_x,
                    rp_y=coord_y
                )
                record_count += 1

            # 步骤2：补新批次
            if record_count < max_num and new_batch_count > 0:
                need_new = max_num - record_count
                use_new_batches = new_batch_list[:need_new]
                for new_bid in use_new_batches:
                    rssi_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
                    target_rssi = base_rssi + rssi_offset

                    ap_wifi_sql += WIFI_SQL_VALUE.format(
                        collection_batch_id=collections_batch_id,
                        request_batch_id=new_bid,
                        ap_id=ap_id,
                        ap_name=ap_name,
                        ap_rssi=target_rssi,
                        rp_x=coord_x,
                        rp_y=coord_y
                    )
                    record_count += 1

            # 处理SQL结尾
            ap_wifi_sql = ap_wifi_sql.rstrip(',\n') + ';'

            # ========== 关键修改2：判断是否为空SQL语句 ==========
            if ap_wifi_sql.strip() == WIFI_SQL_INSERT.rstrip('\n') + ';':
                continue  # 跳过空SQL
            wifi_final_sql += ap_wifi_sql + "\n\n"

        # ===================== 控制台打印 + 写入文件 =====================
        coord_output = f"-- 坐标：({coord_x}, {coord_y})\n"
        # ========== 关键修改3：只有当有有效SQL时才添加对应内容 ==========
        if cell_final_sql:
            coord_output += f"-- Cell AP SQL\n{cell_final_sql}\n\n"
        if wifi_final_sql:
            coord_output += f"-- WiFi AP SQL\n{wifi_final_sql}\n\n"

        # 控制台打印（仅打印有内容的部分）
        if cell_final_sql:
            print(f"-- 坐标 ({coord_x},{coord_y}) Cell AP SQL")
            print(cell_final_sql)
        if wifi_final_sql:
            print(f"-- 坐标 ({coord_x},{coord_y}) WiFi AP SQL")
            print(wifi_final_sql)
        if cell_final_sql or wifi_final_sql:
            print("-" * 80 + "\n")

        # 追加写入文件（仅写入有内容的部分）
        if cell_final_sql or wifi_final_sql:
            with open(sql_output_file, 'a', encoding='utf-8') as f:
                f.write(coord_output)


# def build_cell_sql(fill_map: dict):
#     # 修正WiFi的source字段（原错误写为'cell'）
#     CELL_SQL_INSERT = "insert into single_collection_data (space_id, collection_batch_id, request_batch_id, ap_id, ap_rsrp, ap_rsrq, ap_sinr, rp_x, rp_y, source,type) values\n"
#     CELL_SQL_VALUE = "(15,1772167533230,{request_batch_id},'{ap_id}',{ap_rsrp},{ap_rsrq},{ap_sinr},{rp_x},{rp_y},'cell',2),\n"
#
#     WIFI_SQL_INSERT = "insert into single_collection_data (space_id, collection_batch_id, request_batch_id, ap_id,ap_name, ap_rssi, rp_x, rp_y, source,type) values\n"
#     WIFI_SQL_VALUE = "(15,1772167533230,{request_batch_id},'{ap_id}','{ap_name}',{ap_rssi},{rp_x},{rp_y},'wifi',2),\n"
#
#     sql_output_file = f"fill_all_sql_{int(time.time() * 1000)}.sql"
#     # 初始化文件（清空原有内容，保证每次运行重新生成）
#     with open(sql_output_file, 'w', encoding='utf-8') as f:
#         f.write(f"-- 信号补全SQL文件，生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"-- 空间ID：15，采集批次ID：{collections_batch_id}\n\n")
#
#     for coord, apx in fill_map.items():
#         coord_array = coord.split('_')
#         coord_x = coord_array[0]
#         coord_y = coord_array[1]
#
#         cell_aps = apx['cell']
#         wifi_aps = apx['wifi']
#
#         # 计算坐标点公共批次并集
#         cell_batch_ids = set([batch_id for ap in cell_aps for batch_id in ap['request_batch_id']])
#         wifi_batch_ids = set([batch_id for ap in wifi_aps for batch_id in ap['request_batch_id']])
#         global_batch_ids = cell_batch_ids | wifi_batch_ids
#         global_batch_sorted = sorted(global_batch_ids)
#         global_batch_len = len(global_batch_ids)
#
#         # 生成坐标点级别的统一新批次列表（所有AP共用）
#         new_batch_count = max(0, max_num - global_batch_len)
#         new_batch_list = []
#         if new_batch_count > 0:
#             base_timestamp = int(time.time() * 1000)
#             # 一行式生成随机8~12秒间隔的新批次
#             new_batch_list = [base_timestamp + sum(random.randint(8000, 12000) for _ in range(i)) for i in range(new_batch_count)]
#
#         # ===================== 核心修正：移出缩进，保证始终处理Cell/WiFi =====================
#         # ===================== 处理Cell AP：每个AP独立补全至max_num =====================
#         cell_final_sql = ""
#         for ap in cell_aps:
#             ap_id = ap['ap_id']
#             base_rsrp = ap['target_rsrp']
#             base_rsrq = ap['target_rsrq']
#             base_sinr = ap['target_sinr']
#             own_batch = set(ap['request_batch_id'])
#
#             ap_cell_sql = CELL_SQL_INSERT
#             record_count = 0
#
#             # 步骤1：补公共批次中自身缺失的部分
#             fill_batch = sorted(global_batch_ids - own_batch)
#             for bid in fill_batch:
#                 if record_count >= max_num:
#                     break
#                 # 随机波动
#                 rsrp_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
#                 rsrq_offset = random.choice([-1, 0, 0, 0, 1])
#                 sinr_offset = random.choice([-1, 0, 0, 0, 1])
#                 target_rsrp = base_rsrp + rsrp_offset
#                 target_rsrq = base_rsrq + rsrq_offset
#                 target_sinr = base_sinr + sinr_offset
#
#                 ap_cell_sql += CELL_SQL_VALUE.format(
#                     collection_batch_id=collections_batch_id,
#                     request_batch_id=bid,
#                     ap_id=ap_id,
#                     ap_rsrp=target_rsrp,
#                     ap_rsrq=target_rsrq,
#                     ap_sinr=target_sinr,
#                     rp_x=coord_x,
#                     rp_y=coord_y
#                 )
#                 record_count += 1
#
#             # 步骤2：补新批次（即使new_batch_count=0，也不影响循环逻辑）
#             if record_count < max_num and new_batch_count > 0:
#                 need_new = max_num - record_count
#                 use_new_batches = new_batch_list[:need_new]
#                 for new_bid in use_new_batches:
#                     rsrp_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
#                     rsrq_offset = random.choice([-1, 0, 0, 0, 1])
#                     sinr_offset = random.choice([-1, 0, 0, 0, 1])
#                     target_rsrp = base_rsrp + rsrp_offset
#                     target_rsrq = base_rsrq + rsrq_offset
#                     target_sinr = base_sinr + sinr_offset
#
#                     ap_cell_sql += CELL_SQL_VALUE.format(
#                         collection_batch_id=collections_batch_id,
#                         request_batch_id=new_bid,
#                         ap_id=ap_id,
#                         ap_rsrp=target_rsrp,
#                         ap_rsrq=target_rsrq,
#                         ap_sinr=target_sinr,
#                         rp_x=coord_x,
#                         rp_y=coord_y
#                     )
#                     record_count += 1
#
#             # 处理SQL结尾
#             ap_cell_sql = ap_cell_sql.rstrip(',\n') + ';'
#             cell_final_sql += ap_cell_sql + "\n\n"
#
#         # ===================== 处理WiFi AP：每个AP独立补全至max_num =====================
#         wifi_final_sql = ""
#         for ap in wifi_aps:
#             ap_id = ap['ap_id']
#             ap_name = ap['ap_name']
#             base_rssi = ap['target_rssi']
#             own_batch = set(ap['request_batch_id'])
#
#             ap_wifi_sql = WIFI_SQL_INSERT
#             record_count = 0
#
#             # 步骤1：补公共批次中自身缺失的部分
#             fill_batch = sorted(global_batch_ids - own_batch)
#             for bid in fill_batch:
#                 if record_count >= max_num:
#                     break
#                 rssi_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
#                 target_rssi = base_rssi + rssi_offset
#
#                 ap_wifi_sql += WIFI_SQL_VALUE.format(
#                     collection_batch_id=collections_batch_id,
#                     request_batch_id=bid,
#                     ap_id=ap_id,
#                     ap_name=ap_name,
#                     ap_rssi=target_rssi,
#                     rp_x=coord_x,
#                     rp_y=coord_y
#                 )
#                 record_count += 1
#
#             # 步骤2：补新批次
#             if record_count < max_num and new_batch_count > 0:
#                 need_new = max_num - record_count
#                 use_new_batches = new_batch_list[:need_new]
#                 for new_bid in use_new_batches:
#                     rssi_offset = random.choice([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
#                     target_rssi = base_rssi + rssi_offset
#
#                     ap_wifi_sql += WIFI_SQL_VALUE.format(
#                         collection_batch_id=collections_batch_id,
#                         request_batch_id=new_bid,
#                         ap_id=ap_id,
#                         ap_name=ap_name,
#                         ap_rssi=target_rssi,
#                         rp_x=coord_x,
#                         rp_y=coord_y
#                     )
#                     record_count += 1
#
#             # 处理SQL结尾
#             ap_wifi_sql = ap_wifi_sql.rstrip(',\n') + ';'
#             wifi_final_sql += ap_wifi_sql + "\n\n"
#
#
#         # ===================== 控制台打印 + 写入文件 =====================
#         coord_output = f"-- 坐标：({coord_x}, {coord_y})\n"
#         coord_output += f"-- Cell AP SQL\n{cell_final_sql}\n\n"
#         coord_output += f"-- WiFi AP SQL\n{wifi_final_sql}\n\n"
#
#         # 控制台打印
#         print(f"-- 坐标 ({coord_x},{coord_y}) Cell AP SQL")
#         print(cell_final_sql)
#         print(f"-- 坐标 ({coord_x},{coord_y}) WiFi AP SQL")
#         print(wifi_final_sql)
#         print("-" * 80 + "\n")
#
#         # 追加写入文件
#         with open(sql_output_file, 'a', encoding='utf-8') as f:
#             f.write(coord_output)




if __name__ == "__main__":
    fill_map = build_fill_map()
    build_cell_sql(fill_map)
