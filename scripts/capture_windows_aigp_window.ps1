param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AigpCaptureRect {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr handle, out RECT rect);
    [DllImport("user32.dll")]
    public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
}
"@

$previousDpiContext = [AigpCaptureRect]::SetThreadDpiAwarenessContext([IntPtr](-4))
$process = Get-Process DCGame-Win64-Shipping -ErrorAction Stop |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
$rect = New-Object AigpCaptureRect+RECT
if (-not [AigpCaptureRect]::GetWindowRect($process.MainWindowHandle, [ref]$rect)) {
    throw "GetWindowRect failed for simulator process $($process.Id)"
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -le 0 -or $height -le 0) {
    throw "Simulator window has invalid dimensions ${width}x${height}"
}

$directory = Split-Path -Parent $OutputPath
if ($directory -and -not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.CopyFromScreen(
        $rect.Left,
        $rect.Top,
        0,
        0,
        $bitmap.Size,
        [System.Drawing.CopyPixelOperation]::SourceCopy
    )
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
    if ($previousDpiContext -ne [IntPtr]::Zero) {
        [AigpCaptureRect]::SetThreadDpiAwarenessContext($previousDpiContext) | Out-Null
    }
}
Write-Output (
    "Captured simulator PID {0}, rect ({1},{2})-({3},{4}), {5}x{6}: {7}" -f
    $process.Id, $rect.Left, $rect.Top, $rect.Right, $rect.Bottom,
    $width, $height, $OutputPath
)
