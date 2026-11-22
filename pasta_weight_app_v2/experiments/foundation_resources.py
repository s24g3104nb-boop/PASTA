"""Process-level peak memory on Linux/macOS/Windows, plus optional CUDA bytes."""
import sys


def peak_rss_bytes():
    if sys.platform.startswith('win'):
        import ctypes as ct
        from ctypes import wintypes as wt

        class ProcessMemoryCounters(ct.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [
                (field, ct.c_size_t) for field in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

        kernel = ct.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wt.HANDLE
        info = ProcessMemoryCounters()
        info.cb = ct.sizeof(info)
        get_info = kernel.K32GetProcessMemoryInfo
        get_info.argtypes = [wt.HANDLE, ct.POINTER(ProcessMemoryCounters), wt.DWORD]
        get_info.restype = wt.BOOL
        if not get_info(kernel.GetCurrentProcess(), ct.byref(info), info.cb):
            return None
        return info.PeakWorkingSetSize

    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak*1024


def torch_runtime(needed, device):
    if not needed:
        return None
    try:
        import torch
    except ImportError as exc:
        raise ValueError('Enable the optional local PyTorch research environment') from exc
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('This PyTorch environment cannot access a CUDA device')
    torch.set_num_threads(1)
    return torch


def device_sync(torch, device):
    if torch is not None and device == 'cuda':
        torch.cuda.synchronize()


def memory_report(torch, device):
    result = dict(host_process_peak_rss_bytes=peak_rss_bytes(), device=device,
                  cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None)
    if torch is not None and device == 'cuda':
        result.update(cuda_device=torch.cuda.get_device_name(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result
