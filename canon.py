import os
import sys
import ctypes
from ctypes import POINTER, c_void_p, c_char_p

# Set the EDSDK_PATH environment variable
os.environ['EDSDK_PATH'] = 'C:/Users/break/Documents/Python Scripts/EDSDK/Dll'
edsdk_path = os.environ['EDSDK_PATH']

# Load EDSDK library
edsdk = ctypes.cdll.LoadLibrary(os.path.join(edsdk_path, 'EDSDK.dll'))

# Error checking function
def check_error(error_code, message=None):
    if error_code != 0:
        if message:
            print(f"{message} (error code: {error_code})")
        else:
            print(f"Error (error code: {error_code})")
        sys.exit(1)

# Callback function for camera events
def event_handler(obj, event, ptr):
    pass

# Initialize the SDK
error = edsdk.EdsInitializeSDK()
check_error(error, "Failed to initialize SDK")

# Get the first camera
camera_ptr = c_void_p()
error = edsdk.EdsGetFirstCamera(ctypes.byref(camera_ptr))
check_error(error, "Failed to get first camera")

# Open a session with the camera
error = edsdk.EdsOpenSession(camera_ptr)
check_error(error, "Failed to open session with camera")

# Set the camera to remote control mode
remote_control_mode = ctypes.c_uint32(0x00000000)
error = edsdk.EdsSetPropertyData(
    camera_ptr,
    ctypes.c_uint32(0x00000000),
    ctypes.c_int32(0),
    ctypes.c_uint32(4),
    ctypes.byref(remote_control_mode)
)
check_error(error, "Failed to set remote control mode")

# Register the event handler callback function
handler_ptr = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, c_void_p)(event_handler)
error = edsdk.EdsSetCameraStateEventHandler(camera_ptr, ctypes.c_uint32(0x00000200), handler_ptr, None)
check_error(error, "Failed to set event handler")

# Take a picture
error = edsdk.EdsSendCommand(camera_ptr, ctypes.c_uint32(0x00000000), ctypes.c_int32(0))
check_error(error, "Failed to take picture")

# Close the session with the camera
error = edsdk.EdsCloseSession(camera_ptr)
check_error(error, "Failed to close session with camera")

# Terminate the SDK
error = edsdk.EdsTerminateSDK()
check_error(error, "Failed to terminate SDK")
