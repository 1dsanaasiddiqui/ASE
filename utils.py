"""
Various utility functions
"""
import logging
import os
from os import path
import sys
from datetime import datetime
import time
import pickle

import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler
import torch
import numpy as np

import config

# Quiet logging from matplotlib
logger = logging.getLogger( 'matplotlib' )
logger.setLevel( level = logging.INFO )


def init_logging():
    """
    Initializes logger
    """
    # Set the log format
    log_format = '\n[%(relativeCreated)s %(filename)s:%(funcName)s:%(lineno)d] %(levelname)s - \n\t %(message)s'

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel( level = logging.DEBUG )

    # Remove existing handlers (if any)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler for the log file
    log_filename = os.path.join(
        config.LOG_PATH,
        "log_{0}.log".format( 
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Create a stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)  # Adjust the log level as needed
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)

    # Log the program start
    program_start_time = datetime.now()
    logger.info('Logging to %s', log_filename)

    # Log contents of config
    logger.info("Config has: {}".format( 
        [ (n, v) for n, v in config.__dict__.items() if '__' not in n ]
    ))


def sign_with_zeros( x ):
    """
    Just like torch.sign, but says zero for components close to 0.
    """
    return torch.where( 
            torch.isclose( x, torch.tensor(0.) ),
            torch.tensor(0.),
            torch.sign( x )
    )


# The following code was originally written by Diganta for the Merge Across
# Layers work, and has been ported for use here.
"""
A simple timing module that can track multiple concurrently running timers,
which can be started and stopped multiple times. It stores the total, mean and
max time seen by each timer.
"""
if config.TIMING:
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

    # File to store timing info into
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
        Records the time passed since last start under the given head. 
        Returns the time taken by the last call
        """
        global timer_last_start, timers
        t = time.monotonic() - timer_last_start[ time_head_str ]
        record_val( time_head_str, t )
        return t


    def record_val( time_head_str, val ):
        """
        Records the given value under the given head.
        """
        global timer_last_start, timers

        # Get previous record
        if time_head_str in timers:
            num, tot, _, mx = timers[ time_head_str ]
        else:
            num, tot, _, mx = TimerRecord(0, 0., 0., 0.)

        # Update time dict and store
        num += 1
        tot += val
        mx = mx if mx >= val else val
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

        logging.info("Timing data so far:\n {}".format( 
            tabulate(table, headers=header, tablefmt='github', numalign='left')
        ))
           
else:   # Stub if timing is disabled
    def start_timer():
        pass
    def record_time( head ):
        pass
    def log_time():
        pass

if config.VISUALIZE:


    plt.rc( 'axes', prop_cycle = (
        cycler( linestyle = ['-', '--', ':', '-.' ] ) *
        cycler( color = 
            ['#377eb8', '#ff7f00', '#4daf4a',
             '#f781bf', '#a65628', '#984ea3',
             '#999999', '#e41a1c', '#dede00']
        )
    ))
    
    class Plotter:
        """
        A simple class to record, plot and save data.

        Members:
        data_dict   -   Dict taking plot names -> lists with data
        fname       -   Optional, the filename to save to. Saves to a file with
                        timestapm as name if not given
        seps        -   Vertical seperators, tuple of position and color
        since_save  -   Steps since when last save was done
        """
        def __init__( self, fname = None, load = False ):
            """
            Arguments:
            fname   -   Optional, the filename to save to. Saves to a file with
                        timestapm as name if not given
            load    -   If true, loads data from given file
            """
            
            self.fname = fname
            
            if load:
                assert fname is not None
                with open( fname, 'rb' ) as f:
                    self.data_dict, self.seps = pickle.load( f )

            else:
                self.data_dict = {}
                self.seps = []
                
            self.since_save = 0
                
        def record( self, plot_str, val ):
            """
            Record given value under given plot
            """
            if plot_str not in self.data_dict:
                self.data_dict[ plot_str ] = []
            self.data_dict[ plot_str ].append( val )
            self.since_save += 1
            if self.since_save >= config.PLOT_SAVE_IVAL:
                logging.info( "Saving plot data at given interval" )
                self.since_save = 0
                self.save()

        def record_sep( self, color = 'black' ):
            """
            Record a seperator at current position
            """
            assert len(self.data_dict.values()) > 0
            self.seps.append(
                    ( len( list( self.data_dict.values() )[0] ), color ))
            #if config.DEBUG:
            #    logging.debug("Adding seperator to plot @ {}".format( 
            #        len( list( self.data_dict.values() )[0] ) ))

        def save( self ):
            """
            Saves network, returns filename
            """
            fname = os.path.join(
                config.PLOT_PATH,
                "plot_{0}.plot".format( 
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
            ) if self.fname is None else self.fname
            logging.info( "Saving plot data to:\n {}".format( fname ))
            with open( fname, 'wb' ) as f:
                pickle.dump( (self.data_dict, self.seps), f )
            return fname

        def show( self, title = None, show = True, save = True ):
            """
            Shows all plots, and saves them, if corresponding flags are true. If
            save and show are set to true, also saves the image to
            `fname_fig.png`
            """
            if save:
                fname = self.save()
            
            if show:
                for plot_str, vals in self.data_dict.items():
                    plt.plot( vals, label = plot_str )
                for sep_pos, sep_col in self.seps:
                    plt.axvline( sep_pos, color = sep_col, linewidth = 1 )
                plt.legend()
                if title is not None: plt.title( title )

                if save:
                    logging.info( "Saving plot figure to:\n {}_fig.png".format( 
                        fname ))
                    plt.savefig( "{}_fig.png".format( fname ) )

                if config.SHOW_PLOT:
                    plt.show()




    class HistogramPlotter:
        """
        A simple class to record, plot and save data as histograms

        Members:
        num_bins    -   Number of bins in the histogram
        histograms  -   List of histograms, each is an np array
        bin_edges   -   Edges of the bins for the histogram
        fname       -   Optional, the filename to save to. Saves to a file with
                        timestapm as name if not given
        colormap    -   Name of colormap (from mpl.cm.get_cmap) to use.
        seps        -   Vertical seperators
        since_save  -   Steps since when last save was done
        """
        def __init__( self, 
                num_bins = None, min_val = 0., max_val = 1.,
                fname = None, load = False, 
                colormap = 'viridis', ):
            """
            Arguments:
            num_bins    -   Number of bins in the histogram
            fname       -   Optional, the filename to save to. Saves to a file with
                            timestapm as name if not given
            min,max_val -   Minimum and maximum values expected to be recorded.
                            Ignored if load is true.
            load        -   If true, loads data from given file
            colormap    -   Name of colormap (from mpl.cm.get_cmap) to use.
            """
            self.fname = fname
            self.colormap = colormap
            
            # Load histogram, min and max vals, infer num_bins
            if load:
                assert fname is not None
                with open( fname, 'rb' ) as f:
                    self.histograms, self.seps, min_val, max_val = (
                            pickle.load( f ))
                num_bins = self.histograms[0].shape[0]

            # Initialize histograms, check number of bins, min & man is valid
            else:
                assert num_bins is not None
                assert min_val < max_val
                self.histograms = []
                self.seps = []

            # Create bin_edges
            self.bin_edges = np.linspace( 
                    start = min_val, stop = max_val, num = num_bins + 1 )

            self.since_save = 0
                
        def record( self, vals ):
            """
            Record given value under given plot
            """
            hist, _ = np.histogram( 
                vals, bins = self.bin_edges, density = True )
            self.histograms.append( hist )
            self.since_save += 1
            if self.since_save >= config.PLOT_SAVE_IVAL:
                logging.info( "Saving plot data at given interval" )
                self.since_save = 0
                self.save()

        def record_sep( self, color = 'black' ):
            """
            Record a seperator at current position
            """
            self.seps.append(( len(self.histograms), color ))

        def save( self ):
            fname = os.path.join(
                config.PLOT_PATH,
                "histogram_plot_{0}.plot".format( 
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
            ) if self.fname is None else self.fname
            logging.info( "Saving plot data to:\n {}".format( fname ))
            with open( fname, 'wb' ) as f:
                pickle.dump((
                        self.histograms, 
                        self.seps,
                        self.bin_edges[0], 
                        self.bin_edges[-1],
                    ), f )
            return fname

        def show( self, title = None, show = True, save = True ):
            """
            Shows all plots, and saves them, if corresponding flags are false.
            """

            # Get filename
            if save:
                fname = self.save()
            
            if show:
                img = np.stack( self.histograms ).T
                ys = self.bin_edges
                xs = np.arange( len(self.histograms) + 1,
                        dtype=config.FLOAT_TYPE_NP  )
                plt.pcolormesh( xs, ys, img, 
                        cmap = mpl.cm.get_cmap( self.colormap ), 
                )
                for sep_pos, sep_col in self.seps:
                    plt.axvline( sep_pos, color = sep_col, linewidth = 1 )
                if title is not None: plt.title( title )

                # Save figure
                if save:
                    logging.info( "Saving plot figure to:\n {}_fig.png".format( 
                        fname ))
                    plt.savefig( "{}_fig.png".format( fname ) )

                if config.SHOW_PLOT:
                    plt.show()



def dump( data ):
    """
    Dumps a bunch of data to a dump file. File name is dump followed by
    timestamp.    
    """
    dump_fname = path.join(
        config.DUMP_PATH,
        "dump_{}".format( str( datetime.now() ))
    )
    logging.info("Dumping at {}".format(dump_fname))
    with open( dump_fname, 'wb' ) as f:
        pickle.dump( data, f )


def load_dump( dump_file ):
    """
    Loads a bunch of data from the given dump file
    """
    with open( dump_file , 'rb' ) as f:
        data = pickle.load( f )
    return data


def grad_log_hook( name, idxs = None ):
    """
    Returns a hook that prints gradient with given name, optionally at given
    index
    """
    if idxs is not None:
        def __log_hook( grad ):
            logging.debug( "Grad of {} at {} is {}".format( name, idxs, grad[ idxs ] ))
    else: 
        def __log_hook( grad ):
            logging.debug( "Grad of {} is {}".format( name, grad ))

    return __log_hook


class TableComparator:
    """
    A class to compare and print a bunch of values as a table
    
    Members:
    table   -   Dict taking header to list of vals
    """
    def __init__(self):
        self.table = {}

    def add_col(self, name, data):
        self.table[name] = data

    def show(self):
        logging.debug("Comparing {}".format( self.table.keys() ))
        itms = list( self.table.values() )
        blk = '\n'
        for i in range( itms[0].shape[0] ):   
            ln = ''
            for itm in itms:
                dt = itm[i].tolist()
                if type(dt) is list:
                    st = '[{}]'.format(','.join([
                        '{:.3f}'.format(n) for n in dt ]))
                else:
                    st = '{:.3f}'.format(dt)
                #l.debug( dt )
                #l.debug( type(dt) )
                #l.debug( st ) 
                ln += ' {}'.format( st )
            blk += '\n{}'.format( ln )
        logging.debug("Comparisions: {}".format( blk ))
    

if __name__ == "__main__":
    
    init_logging()

    p = Plotter('bar.plot')
    p.record('foo', 1 )
    p.record('bar', 2 )
    p.record('foo', 3 )
    p.record('bar', 4 )
    #p.show()

    def foo( x ):
        y = x**2
        z = y + y
        logging.info(locals())

    a = 5
    foo( a )
   
    x = torch.tensor( [2., 2.] )
    grad_log_hook( 'x-foo', [1, None] )( x )

def str_with_idx( arr ):
    """
    Converts a flat numpy array / torch tensor to string with indices for each
    element.
    """
    return '\n'.join( ['{} : {}'.format( i, e ) for i, e in enumerate( arr )])
