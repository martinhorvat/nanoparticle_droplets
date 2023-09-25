import socket
from log import create_logger

logger = create_logger(__name__, 'INFO')

class Device:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        pass

    def _query(self, order='', bufferSize=1024, timeout=10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((self.host, self.port))
            logger.debug(f'Sending command "{order}".')
            s.send(f'{order}\r\n'.encode())

            echo = s.recv(bufferSize).decode().strip()
            logger.debug(f'Received error: "{echo}"')
        return echo
    
class Camera(Device):
    def __init__(self, host='127.0.0.1', port=5000):
        super().__init__(host, port)

        # test connection
        connTest = self.query('test')
        logger.info(f'Tested connection on {self.host}:{self.port}. Response = "{connTest}"')
        if not connTest == 'unknown command':
            logger.warn(f'Got "{connTest}" instead of "1".')

    def query(self, order='', bufferSize=1024):
        return self._query(order, bufferSize)


    def setFolder(self, folder=''):
        return self.query(f'SETFOLDER {folder}')

    def takeImage(self, filename=''):
        return self.query(f'SNAPSHOT {filename}.jpg')
    
    def setExposure(self, exposure_ms=200):
        return self.query(f'SET_EXPOSURE {exposure_ms}')