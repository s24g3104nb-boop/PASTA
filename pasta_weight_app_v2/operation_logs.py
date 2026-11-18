"""Local operational events. Callers supply fixed feature and operation text."""
from functools import wraps
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from threading import Lock
from time import gmtime


WEIGHT = '重量推定'
NUTRITION = '栄養計算'
RECORDS = '記録管理'

_LOGGER = logging.getLogger('pasta_lens.operations')
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())
_HANDLER_NAME = 'pasta-lens-local-file'
_CONFIG_LOCK = Lock()


def _report_file_failure():
    # The standard Handler prints a traceback with the failing path by default.
    try:
        print('Pasta Lens: 操作ログの保存先を利用できません。', file=sys.stderr)
    except (OSError, ValueError):
        pass


class _OperationFileHandler(RotatingFileHandler):
    def handleError(self, record):
        _report_file_failure()


def configure_operation_log(data_dir):
    """Use one handler across Streamlit reruns and isolate it from root logging."""
    path = (Path(data_dir) / 'logs' / 'pasta_lens.log').absolute()
    with _CONFIG_LOCK:
        for handler in _LOGGER.handlers.copy():
            if handler.name != _HANDLER_NAME:
                continue
            if handler.baseFilename == str(path):
                return
            _LOGGER.removeHandler(handler)
            handler.close()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = _OperationFileHandler(path, maxBytes=1024 * 1024, backupCount=3, encoding='utf-8')
        except OSError:
            _report_file_failure()
            return
        handler.name = _HANDLER_NAME
        formatter = logging.Formatter('%(asctime)sZ | %(levelname)s | %(feature)s | %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%S')
        formatter.converter = gmtime
        handler.setFormatter(formatter)
        _LOGGER.addHandler(handler)


def log_info(feature, message):
    _LOGGER.info(message, extra={'feature': feature})


def log_error(feature, message):
    _LOGGER.error(message, extra={'feature': feature})


def logged_operation(feature, action, *, report_success=True):
    """Log completed operations and unexpected exceptions without payloads.

    These functions use ValueError for ordinary input and state rejections.
    Invalid JSON from persisted state is a processing failure.
    """
    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if not isinstance(exc, ValueError) or isinstance(exc, json.JSONDecodeError):
                    log_error(feature, f'{action} 失敗')
                raise
            if report_success:
                log_info(feature, f'{action} 完了')
            return result
        return call
    return decorate
