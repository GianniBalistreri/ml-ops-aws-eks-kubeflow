"""

Customized logger

"""

import os

from datetime import datetime
from typing import List


class Log:
    """
    Class for handling logging
    """
    def __init__(self,
                 write: bool = False,
                 level: str = 'info',
                 env: str = 'dev',
                 logger_file_path: str = None,
                 log_ram_usage: bool = True,
                 log_cpu_usage: bool = True
                 ):
        """
        :param write: bool
            Write logging file or not

        :param level: str
            Name of the logging level of the messge
                -> info: Logs any information
                -> warn: Logs warnings
                -> error: Logs errors including critical messages

        :param env: str
            Name of the logging environment to use
                -> dev: Development - Logs any information
                -> stage: Staging - Logs only warnings and errors including critical messages
                -> prod: Production - Logs only errors including critical messages

        :param logger_file_path: str
            Complete file path of the logger file

        :param log_ram_usage: bool
            Log RAM usage (in percent) or not

        :param log_cpu_usage: bool
            Log CPU usage (in percent) or not
        """
        self.write: bool = write
        self.timestamp_format: str = '%Y-%m-%d %H:%M:%S'
        if log_ram_usage:
            #self.ram: str = ' -> RAM {}%'.format(psutil.virtual_memory().percent)
            self.ram: str = ''
        else:
            self.ram: str = ''
        if log_cpu_usage:
            #self.cpu: str = ' -> CPU {}%'.format(psutil.cpu_percent(percpu=False))
            self.cpu: str = ''
        else:
            self.cpu: str = ''
        self.msg: str = f'{datetime.now().strftime(self.timestamp_format)}{self.ram}{self.cpu} | '
        if write:
            if logger_file_path is None:
                self.log_file_path: str = os.path.join(os.getcwd(), 'log.txt')
            else:
                self.log_file_path: str = logger_file_path
        else:
            self.log_file_path: str = None
        self.levels: List[str] = ['info', 'warn', 'error']
        _env: dict = dict(dev=0, stage=1, prod=2)
        if env in _env.keys():
            self.env: int = _env.get(env)
        else:
            self.env: int = _env.get('dev')
        if level in self.levels:
            self.level: int = self.levels.index(level)
        else:
            self.level: int = 0

    def _write(self):
        """
        Write log file
        """
        with open(file=self.log_file_path, mode='a', encoding='utf-8') as _log:
            _log.write('{}\n'.format(self.msg))

    def log(self, msg: str):
        """
        Log message

        :param msg: str
            Message to log
        """
        if self.level >= self.env:
            if self.level == 0:
                _level_description: str = ''
            elif self.level == 1:
                _level_description: str = 'WARNING: '
            elif self.level == 2:
                _level_description: str = 'ERROR: '
            self.msg = f'{self.msg}{msg}'
            if self.write:
                self._write()
            else:
                print(self.msg)
