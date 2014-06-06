
class PrivateData(object):
    """Represent the private information of a subject
    """
    def __init__(self):
        self.name
        self.birthdate

class Subject(object):
    """Represent the subject information of a
        population study
    """
    def __init__(self):
        self.private = PrivateData()
        self.private.name = ''
        self.private.birthdate = ''

        self.id
        self.images