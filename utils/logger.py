import logging
from datetime import datetime

# 创建一个日志格式化器
def custom_formatter():
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    return logging.Formatter(fmt, datefmt)

# 创建一个用于文件日志的处理器
def file_handler(log_file):
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)  # 设置文件处理器的级别
    handler.setFormatter(custom_formatter())
    return handler

# 创建一个用于控制台日志的处理器
def stream_handler():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # 设置控制台处理器的级别
    handler.setFormatter(custom_formatter())
    return handler

# 创建一个配置好的日志器
def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志器的级别

    if log_file:
        file_hdlr = file_handler(log_file)
        logger.addHandler(file_hdlr)

    stream_hdlr = stream_handler()
    logger.addHandler(stream_hdlr)

    return logger

# 使用示例
if __name__ == "__main__":
    log_file = "application.log"  # 根据需要更改日志文件路径
    logger = get_logger("my_app", log_file)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")