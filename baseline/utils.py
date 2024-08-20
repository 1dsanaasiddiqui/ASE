import os
from os import path
import sys
from datetime import datetime
import time
import pickle

from matplotlib import pyplot as plt
import torch

from collections import namedtuple 
from tabulate import tabulate

# A named tuple that stores the number of records, total time, mean time,
# and max time for each timer
TimerRecord = namedtuple( 'TimerRecord', [
    'num_records', 
    'total', 
    'mean',
    'max'
])

# Dict taking timer string -> time at which it was last started.
timer_last_start = {}

# Dict taking timer string -> TimerRecord
timers = {}

time_file_name = None

def start_timer( time_head_str ):
        """
        Starts a timer under the given head. The timer's current value can be
        recorded with `record_time` calls.
        """
        global timer_val
        timer_last_start[ time_head_str ] = time.monotonic()

def record_time( time_head_str ):
        """
        Records the time passed since last start under the given head. The dict
        `timers` stores a tuple of the following things:
        1.  Total time accumulated under the head
        2.  List of individual time records

        Returns:
        The time taken by the last call
        """
        global timer_last_start, timers

        # Get time
        t = time.monotonic() - timer_last_start[ time_head_str ]

        # Get previous record
        if time_head_str in timers:
            num, tot, _, mx = timers[ time_head_str ]
        else:
            num, tot, _, mx = TimerRecord(0, 0., 0., 0.)

        # Update time dict and store
        num += 1
        tot += t
        mx = mx if mx >= t else t
        timers[ time_head_str ] = TimerRecord( num, tot, tot / num, mx )

        # Dump out time dict
        if time_file_name is not None:
            with open( time_file_name, 'wb' ) as f:
                pickle.dump( timers, f )
def log_times():
        """
        Logs the time collected so far for each head. For each head, prints the
        following:
        1.  Total time taken
        2.  Mean time
        3.  Max time
        4.  Percentage of calls within 5% of maximum
        5.  Total number of calls
        """

        header = ["Timer", "Mean", "Max", "Total", "No. Calls"]
        table = []
        for head, record in timers.items():
            table.append([ 
                head, record.mean, record.max, record.total, record.num_records
            ])

        print("Timing data so far:\n {}".format( 
            tabulate(table, headers=header, tablefmt='github')
        ))
