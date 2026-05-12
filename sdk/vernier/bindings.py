import platform
import ctypes

VERNIER_DEFAULT_VENDOR_ID = 0x08F7
USB_DIRECT_TEMP_DEFAULT_PRODUCT_ID = 0x0002
SKIP_DEFAULT_PRODUCT_ID = 0x0003
CYCLOPS_DEFAULT_PRODUCT_ID = 0x0004
MINI_GC_DEFAULT_PRODUCT_ID = 0x0007
SKIP_CMD_ID_START_MEASUREMENTS = 0x18
SKIP_CMD_ID_STOP_MEASUREMENTS = 0x19

class GoIOSDK:
    def __init__(self):
        self._lib = None
        self._load_library()

    def _load_library(self):
        system = platform.system()
        if system == 'Windows':
            self._lib = ctypes.CDLL('./sdk/vernier/win/GoIO64.dll')
        elif system == 'Darwin':
            if platform.machine() == 'arm64':
                self._lib = ctypes.CDLL('./sdk/vernier/mac/libGoIOUniversal.dylib')
            else:
                self._lib = ctypes.CDLL('./sdk/vernier/mac/libGoIO64.dylib')
        elif system == 'Linux':
            self._lib = ctypes.CDLL('./sdk/vernier/linux/libGoIO64.so')

        self._lib.GoIO_Init.restype = ctypes.c_int
        self._lib.GoIO_Uninit.restype = ctypes.c_int

        self._lib.GoIO_GetDLLVersion.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16)]
        self._lib.GoIO_GetDLLVersion.restype = ctypes.c_int

        self._lib.GoIO_UpdateListOfAvailableDevices.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.GoIO_UpdateListOfAvailableDevices.restype = ctypes.c_int

        self._lib.GoIO_GetNthAvailableDeviceName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                             ctypes.c_int]
        self._lib.GoIO_GetNthAvailableDeviceName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_Open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.GoIO_Sensor_Open.restype = ctypes.c_void_p

        self._lib.GoIO_Sensor_Close.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_Close.restype = ctypes.c_int

        self._lib.GoIO_Sensor_GetOpenDeviceName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self._lib.GoIO_Sensor_GetOpenDeviceName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_Lock.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.GoIO_Sensor_Lock.restype = ctypes.c_int

        self._lib.GoIO_Sensor_Unlock.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_Unlock.restype = ctypes.c_int

        self._lib.GoIO_Sensor_ClearIO.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_ClearIO.restype = ctypes.c_int

        self._lib.GoIO_Sensor_SendCmdAndGetResponse.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_void_p,
                                                                ctypes.c_uint, ctypes.c_void_p,
                                                                ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
        self._lib.GoIO_Sensor_SendCmdAndGetResponse.restype = ctypes.c_int

        self._lib.GoIO_Sensor_GetLastCmdResponseStatus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
                                                                   ctypes.POINTER(ctypes.c_ubyte),
                                                                   ctypes.POINTER(ctypes.c_ubyte),
                                                                   ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_GetLastCmdResponseStatus.restype = ctypes.c_int

        self._lib.GoIO_Sensor_GetMeasurementTickInSeconds.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetMeasurementTickInSeconds.restype = ctypes.c_double

        self._lib.GoIO_Sensor_GetMinimumMeasurementPeriod.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetMinimumMeasurementPeriod.restype = ctypes.c_double

        self._lib.GoIO_Sensor_GetMaximumMeasurementPeriod.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetMaximumMeasurementPeriod.restype = ctypes.c_double

        self._lib.GoIO_Sensor_SetMeasurementPeriod.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
        self._lib.GoIO_Sensor_SetMeasurementPeriod.restype = ctypes.c_int

        self._lib.GoIO_Sensor_GetMeasurementPeriod.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.GoIO_Sensor_GetMeasurementPeriod.restype = ctypes.c_double

        self._lib.GoIO_Sensor_GetNumMeasurementsAvailable.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetNumMeasurementsAvailable.restype = ctypes.c_int

        self._lib.GoIO_Sensor_ReadRawMeasurements.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                                              ctypes.c_uint]
        self._lib.GoIO_Sensor_ReadRawMeasurements.restype = ctypes.c_int

        self._lib.GoIO_Sensor_GetLatestRawMeasurement.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetLatestRawMeasurement.restype = ctypes.c_int

        self._lib.GoIO_Sensor_ConvertToVoltage.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.GoIO_Sensor_ConvertToVoltage.restype = ctypes.c_double

        self._lib.GoIO_Sensor_CalibrateData.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self._lib.GoIO_Sensor_CalibrateData.restype = ctypes.c_double

        self._lib.GoIO_Sensor_GetProbeType.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_GetProbeType.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_ReadRecord.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.GoIO_Sensor_DDSMem_ReadRecord.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_WriteRecord.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.GoIO_Sensor_DDSMem_WriteRecord.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.GoIO_Sensor_DDSMem_SetRecord.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.GoIO_Sensor_DDSMem_GetRecord.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_ClearRecord.argtypes = [ctypes.c_void_p]
        self._lib.GoIO_Sensor_DDSMem_ClearRecord.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_CalculateChecksum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_CalculateChecksum.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetMemMapVersion.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetMemMapVersion.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetMemMapVersion.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetMemMapVersion.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetSensorNumber.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetSensorNumber.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetSensorNumber.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
                                                                 ctypes.c_int, ctypes.c_int]
        self._lib.GoIO_Sensor_DDSMem_GetSensorNumber.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetSerialNumber.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_ubyte,
                                                                 ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetSerialNumber.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetSerialNumber.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
                                                                 ctypes.POINTER(ctypes.c_ubyte),
                                                                 ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetSerialNumber.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetLotCode.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetLotCode.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetLotCode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
                                                            ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetLotCode.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetManufacturerID.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetManufacturerID.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetManufacturerID.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetManufacturerID.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetLongName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.GoIO_Sensor_DDSMem_SetLongName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetLongName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint16]
        self._lib.GoIO_Sensor_DDSMem_GetLongName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetShortName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.GoIO_Sensor_DDSMem_SetShortName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetShortName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint16]
        self._lib.GoIO_Sensor_DDSMem_GetShortName.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetUncertainty.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetUncertainty.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetUncertainty.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetUncertainty.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetSignificantFigures.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetSignificantFigures.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetSignificantFigures.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetSignificantFigures.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetCurrentRequirement.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetCurrentRequirement.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetCurrentRequirement.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetCurrentRequirement.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetAveraging.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetAveraging.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetAveraging.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetAveraging.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetMinSamplePeriod.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.GoIO_Sensor_DDSMem_SetMinSamplePeriod.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetMinSamplePeriod.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._lib.GoIO_Sensor_DDSMem_GetMinSamplePeriod.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetTypSamplePeriod.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.GoIO_Sensor_DDSMem_SetTypSamplePeriod.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetTypSamplePeriod.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._lib.GoIO_Sensor_DDSMem_GetTypSamplePeriod.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetTypNumberofSamples.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
        self._lib.GoIO_Sensor_DDSMem_SetTypNumberofSamples.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetTypNumberofSamples.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
        self._lib.GoIO_Sensor_DDSMem_GetTypNumberofSamples.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetWarmUpTime.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
        self._lib.GoIO_Sensor_DDSMem_SetWarmUpTime.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetWarmUpTime.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
        self._lib.GoIO_Sensor_DDSMem_GetWarmUpTime.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetExperimentType.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetExperimentType.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetExperimentType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetExperimentType.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetOperationType.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetOperationType.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetOperationType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetOperationType.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetCalibrationEquation.argtypes = [ctypes.c_void_p, ctypes.c_char]
        self._lib.GoIO_Sensor_DDSMem_SetCalibrationEquation.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetCalibrationEquation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self._lib.GoIO_Sensor_DDSMem_GetCalibrationEquation.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetYminValue.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.GoIO_Sensor_DDSMem_SetYminValue.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetYminValue.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._lib.GoIO_Sensor_DDSMem_GetYminValue.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetYmaxValue.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.GoIO_Sensor_DDSMem_SetYmaxValue.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetYmaxValue.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._lib.GoIO_Sensor_DDSMem_GetYmaxValue.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetYscale.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetYscale.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetYscale.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetYscale.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetHighestValidCalPageIndex.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetHighestValidCalPageIndex.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetHighestValidCalPageIndex.argtypes = [ctypes.c_void_p,
                                                                             ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetHighestValidCalPageIndex.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetActiveCalPage.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetActiveCalPage.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetActiveCalPage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetActiveCalPage.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetCalPage.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_float,
                                                            ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        self._lib.GoIO_Sensor_DDSMem_SetCalPage.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetCalPage.argtypes = [ctypes.c_void_p, ctypes.c_ubyte,
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float), ctypes.c_char_p,
                                                            ctypes.c_uint16]
        self._lib.GoIO_Sensor_DDSMem_GetCalPage.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_SetChecksum.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        self._lib.GoIO_Sensor_DDSMem_SetChecksum.restype = ctypes.c_int

        self._lib.GoIO_Sensor_DDSMem_GetChecksum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        self._lib.GoIO_Sensor_DDSMem_GetChecksum.restype = ctypes.c_int

        self._lib.GoIO_Diags_SetDebugTraceThreshold.argtypes = [ctypes.c_int]
        self._lib.GoIO_Diags_SetDebugTraceThreshold.restype = ctypes.c_int

        self._lib.GoIO_Diags_GetDebugTraceThreshold.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.GoIO_Diags_GetDebugTraceThreshold.restype = ctypes.c_int

    def get_device(self):
        max_path = 256
        device_name = ctypes.create_string_buffer(max_path)
        vendor_id = ctypes.c_int()
        product_id = ctypes.c_int()
        found = False

        num_skips = self._lib.GoIO_UpdateListOfAvailableDevices(VERNIER_DEFAULT_VENDOR_ID, SKIP_DEFAULT_PRODUCT_ID)
        num_temps = self._lib.GoIO_UpdateListOfAvailableDevices(VERNIER_DEFAULT_VENDOR_ID,
                                                                USB_DIRECT_TEMP_DEFAULT_PRODUCT_ID)
        num_cyclopses = self._lib.GoIO_UpdateListOfAvailableDevices(VERNIER_DEFAULT_VENDOR_ID,
                                                                    CYCLOPS_DEFAULT_PRODUCT_ID)
        num_minigcs = self._lib.GoIO_UpdateListOfAvailableDevices(VERNIER_DEFAULT_VENDOR_ID, MINI_GC_DEFAULT_PRODUCT_ID)

        if num_skips > 0:
            self._lib.GoIO_GetNthAvailableDeviceName(device_name, max_path, VERNIER_DEFAULT_VENDOR_ID,
                                                     SKIP_DEFAULT_PRODUCT_ID, 0)
            vendor_id.value = VERNIER_DEFAULT_VENDOR_ID
            product_id.value = SKIP_DEFAULT_PRODUCT_ID
            found = True
        elif num_temps > 0:
            self._lib.GoIO_GetNthAvailableDeviceName(device_name, max_path, VERNIER_DEFAULT_VENDOR_ID,
                                                     USB_DIRECT_TEMP_DEFAULT_PRODUCT_ID, 0)
            vendor_id.value = VERNIER_DEFAULT_VENDOR_ID
            product_id.value = USB_DIRECT_TEMP_DEFAULT_PRODUCT_ID
            found = True
        elif num_cyclopses > 0:
            self._lib.GoIO_GetNthAvailableDeviceName(device_name, max_path, VERNIER_DEFAULT_VENDOR_ID,
                                                     CYCLOPS_DEFAULT_PRODUCT_ID, 0)
            vendor_id.value = VERNIER_DEFAULT_VENDOR_ID
            product_id.value = CYCLOPS_DEFAULT_PRODUCT_ID
            found = True
        elif num_minigcs > 0:
            self._lib.GoIO_GetNthAvailableDeviceName(device_name, max_path, VERNIER_DEFAULT_VENDOR_ID,
                                                     MINI_GC_DEFAULT_PRODUCT_ID, 0)
            vendor_id.value = VERNIER_DEFAULT_VENDOR_ID
            product_id.value = MINI_GC_DEFAULT_PRODUCT_ID
            found = True

        return found, device_name.value.decode(), vendor_id.value, product_id.value

    def get_library(self):
        return self._lib
