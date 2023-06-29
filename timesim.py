import numpy as np
import random
import copy


class Timesim:
    def __init__(self, per_round_time_range):
        self.per_round_time_range = per_round_time_range

        self._cur_time = 0
        self.clients_per_round_time = {}
        self.events = []

    def set_time(self, time):
        self._cur_time = time

    def get_time(self):
        return self._cur_time

    def setup_clients_per_round_time(self, clients):
        for c in clients:
            randfloat = random.random()

            self.clients_per_round_time[c.id] = self.per_round_time_range[
                0
            ] + randfloat * (
                self.per_round_time_range[1] - self.per_round_time_range[0]
            )

    def register_train_events(self, clients, func, base_params):
        random.shuffle(clients)
        for c in clients:
            end_time = self._cur_time + self.clients_per_round_time[c.id]

            params = copy.deepcopy(base_params)
            params["client"] = c

            self.events.append((end_time, func, params))

        self.events = sorted(self.events, key=lambda k: k[0])

    def pop_events(self, count):
        available_events_cnt = min(count, len(self.events))
        event_popped_clients = []
        for i in range(available_events_cnt):
            event = self.events.pop(0)
            event[1](**event[2])
            self._cur_time = event[0]
            event_popped_clients.append(event[2]["client"])
        return available_events_cnt, event_popped_clients
