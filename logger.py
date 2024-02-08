import os
import logging

# creating a logger 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# creating a filehandler and formatter, append the format to the file
handler=logging.FileHandler('sample_file.log', mode='w')
formatter = logging.Formatter("%(asctime)s - %(module)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# adding handler to the logger
logger.addHandler(handler)

# logging
logger.info('debugging message')