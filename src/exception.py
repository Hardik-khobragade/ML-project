import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.logger import logging 

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    
    error_message="Error occured in python script [{0}] in line number [{1}] error message is[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class Custom_exception_handling(Exception):
    def __init__(self,error_message,error_details):
        super().__init__(error_message)
        self.error_message=error_message_detail(error=error_message,error_detail=error_details)
    
    def __str__(self):
        return self.error_message
    
if __name__ == '__main__':
    try:
        a=1/0
    except Exception as e:
        logging.info('This is divide by zero error')
        raise Custom_exception_handling(e,sys)
    
