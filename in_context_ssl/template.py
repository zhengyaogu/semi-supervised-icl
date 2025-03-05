from string import Template

class ABCTemplate:
    def __init__(self):
        raise NotImplementedError()
    
    def format(self, content: dict) -> str:
        return self.template.substitute(**content)

class ABCInstanceTemplate(ABCTemplate):

    def __init__(self, connector: str):
        self.connector = connector