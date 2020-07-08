
class car_input:
    """ Car control input from control or human
    """

    def __init__(self):
        self.steering=0
        self.throttle=0
        self.brake=0

    def __str__(self):
        return 'steering={:.2f}, throttle={:.2f}, brake={:.2f}'.format(self.steering,self.throttle,self.brake)
