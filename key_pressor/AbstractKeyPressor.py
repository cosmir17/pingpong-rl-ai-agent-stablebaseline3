import abc

class AbstractKeyPressor(abc.ABC):
    @property
    @abc.abstractmethod
    def mouse_click(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def mouse_up(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def mouse_down(self):
        return 'Should never reach here'