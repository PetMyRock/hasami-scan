"""The pytwain structures the helper uses (same layout as pytwain 2.3.0)."""
import ctypes as ct


class TW_VERSION(ct.Structure):
    _pack_ = 2
    _fields_ = [("MajorNum", ct.c_uint16), ("MinorNum", ct.c_uint16), ("Language", ct.c_uint16),
                ("Country", ct.c_uint16), ("Info", ct.c_char * 34)]


class TW_IDENTITY(ct.Structure):
    _pack_ = 2
    _fields_ = [("Id", ct.c_uint32), ("Version", TW_VERSION), ("ProtocolMajor", ct.c_uint16),
                ("ProtocolMinor", ct.c_uint16), ("SupportedGroups", ct.c_uint32),
                ("Manufacturer", ct.c_char * 34), ("ProductFamily", ct.c_char * 34),
                ("ProductName", ct.c_char * 34)]
