#!/bin/bash
# Reuse the user's existing NVIDIA userspace graphics libraries, matching the
# cluster kernel driver; no system installation is required.
export G05_NVIDIA_GRAPHICS_ROOT="${G05_NVIDIA_GRAPHICS_ROOT:-/public/node03/users/panbk/data/nvidia-userlibs/580.178.04-1ubuntu1/root}"
if ! rg -q '580\.178\.04' /proc/driver/nvidia/version; then
    echo 'NVIDIA graphics library / kernel driver version mismatch' >&2
    return 1
fi
test -f "$G05_NVIDIA_GRAPHICS_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json" || return 1
export LD_LIBRARY_PATH="$G05_NVIDIA_GRAPHICS_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$G05_NVIDIA_GRAPHICS_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
