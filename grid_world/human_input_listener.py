import threading


class HumanInputListener:
    def __init__(self):
        self.input_queue = []
        self.input_buf_lock = threading.Lock()

    def buf_human_input(self, human_input):
        with self.input_buf_lock:
            self.input_queue.append(human_input)

    def retrieve_human_input(self):
        with self.input_buf_lock:
            if len(self.input_queue) == 0:
                return None
            else:
                # take the first element in the list
                human_input = self.input_queue.pop()
                return human_input
