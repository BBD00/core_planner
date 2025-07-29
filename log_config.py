import os
from loguru import logger

def setup_logger(worker_id=None):
    """配置loguru日志设置"""
    logger.remove()  # 移除默认处理程序
    
    log_file = os.environ.get("LOG_FILE_PATH", "app.log")
    logger.add(log_file, level="INFO")
    
    return logger

# 预配置一个logger实例
logger = setup_logger()