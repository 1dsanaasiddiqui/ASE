"""
Main file
"""
import logging

import config
import utils



if __name__ == "__main__":

    utils.init_logging()

    try:
        # Rest of code
        logging.info("Starting code")
    
    # Print traceback
    except Exception as e:
        logging.error("Fatal error:", exc_info = True)
        raise e
        
        
