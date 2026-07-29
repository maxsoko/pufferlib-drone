param(
    [Parameter(Mandatory = $true)]
    [int]$X,
    [Parameter(Mandatory = $true)]
    [int]$Y,
    [int]$AfterClickMs = 500
)

$ErrorActionPreference = "Stop"
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AigpClick {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr handle);
    [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint x, uint y, uint data, UIntPtr extra);
}
"@

$process = Get-Process DCGame-Win64-Shipping -ErrorAction Stop |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
[AigpClick]::SetForegroundWindow($process.MainWindowHandle) | Out-Null
$previousDpiContext = [AigpClick]::SetThreadDpiAwarenessContext([IntPtr](-4))
try {
    [AigpClick]::SetCursorPos($X, $Y) | Out-Null
    Start-Sleep -Milliseconds 100
    [AigpClick]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 100
    [AigpClick]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
    if ($AfterClickMs -gt 0) {
        Start-Sleep -Milliseconds $AfterClickMs
    }
} finally {
    if ($previousDpiContext -ne [IntPtr]::Zero) {
        [AigpClick]::SetThreadDpiAwarenessContext($previousDpiContext) | Out-Null
    }
}
Write-Output "Clicked simulator PID $($process.Id) at physical screen ($X,$Y)"
