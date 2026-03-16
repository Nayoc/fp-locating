import math

from flask import Flask, request, jsonify

import build
import db.mapper as mapper
import train
import locate
import threading
import logging

# 初始化Flask应用
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('train_thread.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



# # ------------------------------
# # GET接口示例 - 获取资源
# # ------------------------------
# @app.route('/api/resources', methods=['GET'])
# def get_resources():
#     """获取资源列表接口"""
#     try:
#         # 1. 获取请求参数（查询字符串）
#         # 示例：/api/resources?page=1&status=active
#         page = request.args.get('page', 1, type=int)
#         status = request.args.get('status', 'all')
#
#         # 2. 业务逻辑处理（请在这里填充你的代码）
#         # TODO: 实现数据查询、处理等逻辑
#         # 临时模拟数据
#         data = [
#             {"id": 1, "name": "资源1", "status": "active"},
#             {"id": 2, "name": "资源2", "status": "inactive"}
#         ]
#
#         # 3. 构造响应
#         return jsonify({
#             "success": True,
#             "message": "查询成功",
#             "data": data,
#             "page": page,
#             "total": len(data)
#         }), 200
#
#     except Exception as e:
#         # 错误处理
#         return jsonify({
#             "success": False,
#             "message": f"查询失败: {str(e)}"
#         }), 500


# ------------------------------
# POST接口示例 - 创建资源
# ------------------------------
@app.route('/data/build', methods=['POST'])
def build_dataset():
    """构建数据集"""
    try:
        # 1. 获取请求数据（JSON格式）
        # 示例请求体: {"name": "新资源", "description": "这是一个新资源"}
        request_data = request.get_json()

        # 2. 数据验证（请根据实际需求完善）
        if not request_data:
            return jsonify({
                "success": False,
                "message": "请求数据不能为空"
            }), 400

        space_id = request_data['spaceId']
        dataset_id = request_data['datasetId']
        batch_id = request_data['batchId']
        model = request_data['model']

        def run_async(space_id, dataset_id):
            try:
                # 构建数据集
                dir_name = build.run(space_id, batch_id, model=model)
                # 训练数据
                train.run(space_id, dataset_id, dir_name)
            except Exception as e:
                # 子线程异常必须捕获并记录，否则会静默失败
                logger.error(
                    f"异步训练任务失败（spaceId: {space_id}, datasetId: {dataset_id}）",
                    exc_info=True  # exc_info=True 会记录完整的异常堆栈，方便排查
                )

        # 5. 启动子线程执行训练（daemon=True：主线程退出时子线程自动退出，避免僵尸线程）
        train_thread = threading.Thread(
            target=run_async,
            args=(space_id, dataset_id),
            name=f"TrainThread-{dataset_id}",
        )
        train_thread.start()

        # 4. 构造响应
        return jsonify({
            "code": 1,
            "message": "success",
            "data": "success"
        }), 200

    except Exception as e:
        # 错误处理
        return jsonify({
            "success": False,
            "message": f"error: {str(e)}"
        }), 500


@app.route('/data/train', methods=['POST'])
def train_dataset():
    """构建数据集（异步执行训练任务）"""
    try:
        # 1. 解析并验证请求参数
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "success": True,
                "message": "请求数据不能为空"
            }), 400

        dataset_id = request_data.get('datasetId')
        if not dataset_id:
            return jsonify({
                "success": True,
                "message": "缺少必填参数：datasetId"
            }), 400

        dataset = mapper.select_fingerprint_dataset(dataset_id)
        if not dataset:
            return jsonify({
                "success": True,
                "message": f"数据集不存在（datasetId: {dataset_id}）"
            }), 404

        dir_name = dataset.get('dataset_url')
        space_id = dataset.get('space_id')
        space = mapper.select_space(space_id)
        if not space:
            return jsonify({
                "success": True,
                "message": f"空间不存在（spaceId: {space_id}）"
            }), 404

        # 3. 计算norm_y（主线程同步处理参数）
        scale_x = space.get('scale_x')
        scale_rate = space.get('scale_rate')
        if scale_x is None or scale_rate is None:
            return jsonify({
                "success": True,
                "message": "空间配置缺失scale_x或scale_rate"
            }), 400

        norm_y = math.ceil(scale_x if scale_rate > 1 else scale_x / scale_rate)
        space_id = space.get('id')

        # 4. 定义子线程执行的训练函数（包含异常捕获，避免子线程崩溃无日志）
        def run_train_async(space_id, dataset_id, dir_name):
            try:
                train.run(space_id, dataset_id, dir_name)
            except Exception as e:
                # 子线程异常必须捕获并记录，否则会静默失败
                logger.error(
                    f"异步训练任务失败（spaceId: {space_id}, datasetId: {dataset_id}）",
                    exc_info=True  # exc_info=True 会记录完整的异常堆栈，方便排查
                )

        # 5. 启动子线程执行训练（daemon=True：主线程退出时子线程自动退出，避免僵尸线程）
        train_thread = threading.Thread(
            target=run_train_async,
            args=(space_id, dataset_id, dir_name, norm_y),
            name=f"TrainThread-{dataset_id}",
        )
        train_thread.start()

        # 6. 主线程直接返回成功（无需等待子线程完成）
        return jsonify({
            "success": True,
            "message": "success",
            "data": "success"
        }), 200

    except Exception as e:
        # 主线程（参数校验、数据库查询阶段）的异常捕获
        logger.error("训练任务启动失败（主线程异常）", exc_info=True)
        return jsonify({
            "success": True,
            "message": f"训练任务启动失败：{str(e)}"
        }), 500

@app.route('/data/locate', methods=['POST'])
def location():
    try:
        # 1. 解析并验证请求参数
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "code": 400,
                "message": "请求数据不能为空"
            }), 400

        dataset_url = request_data.get('datasetUrl')
        model_file = request_data.get('modelFile')
        ap_list = request_data.get('apList')
        if model_file is None or ap_list is None:
            return jsonify({
                "code": 400,
                "message": "缺少必填参数：datasetId"
            }), 400

        x,y = locate.run(dataset_url,ap_list,model_file)

        # 6. 主线程直接返回成功（无需等待子线程完成）
        return jsonify({
            "code": 1,
            "message": "success",
            "data": {
                "x":x,
                "y":y
            }
        }), 200

    except Exception as e:
        logger.error("定位失败", exc_info=True)
        return jsonify({
            "code": 500,
            "message": f"定位失败：{str(e)}"
        }), 500

# 添加测试接口
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"code": 1, "message": "服务正常运行"}), 200


# 启动服务
if __name__ == '__main__':
    # 开发环境使用debug=True，生产环境需改为False
    app.run(host='0.0.0.0', port=50550, debug=False)
