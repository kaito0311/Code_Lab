import win32api
import sys
import win32gui

def get_active_app():
    '''
    Take app which is running in foreground 
    '''
    active_app = None
    if sys.platform in ['Windows', 'win32']:
        window = win32gui.GetForegroundWindow()
        active_app = win32gui.GetWindowText(window)
    else:
        print("sys.platform = {platform} is not supported".format(
            platform=sys.platform))
        print(sys.version)

    return active_app
def annoucemnt(str):
    win32api.MessageBox(0, f"===================================\n{str}\n=================================")



    
