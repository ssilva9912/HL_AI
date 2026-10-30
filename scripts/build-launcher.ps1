# Builds dist\HomelabAI.exe (with a generated icon) and a Desktop shortcut. No extra tools needed.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $ProjectRoot "dist"
$IcoPath = Join-Path $Dist "homelab-ai.ico"
$ExePath = Join-Path $Dist "HomelabAI.exe"
$Csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"

New-Item -ItemType Directory -Force $Dist | Out-Null

function New-IconPng([int]$Size) {
    $bmp = New-Object System.Drawing.Bitmap($Size, $Size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = "AntiAlias"
    $g.TextRenderingHint = "AntiAlias"

    $radius = [int]($Size * 0.22)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $radius * 2
    $path.AddArc(0, 0, $d, $d, 180, 90)
    $path.AddArc($Size - $d - 1, 0, $d, $d, 270, 90)
    $path.AddArc($Size - $d - 1, $Size - $d - 1, $d, $d, 0, 90)
    $path.AddArc(0, $Size - $d - 1, $d, $d, 90, 90)
    $path.CloseFigure()

    $rect = New-Object System.Drawing.Rectangle(0, 0, $Size, $Size)
    $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        $rect, [System.Drawing.Color]::FromArgb(79, 70, 229), [System.Drawing.Color]::FromArgb(6, 182, 212), 45)
    $g.FillPath($brush, $path)

    $font = New-Object System.Drawing.Font("Segoe UI", [single]($Size * 0.42), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = "Center"
    $format.LineAlignment = "Center"
    $g.DrawString("AI", $font, [System.Drawing.Brushes]::White, (New-Object System.Drawing.RectangleF(0, 0, $Size, $Size)), $format)
    $g.Dispose()

    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    return , $ms.ToArray()
}

# Write a multi-size .ico whose images are embedded PNGs.
$sizes = 16, 32, 48, 64, 128, 256
$images = $sizes | ForEach-Object { , (New-IconPng $_) }
$stream = [System.IO.File]::Create($IcoPath)
$writer = New-Object System.IO.BinaryWriter($stream)
$writer.Write([uint16]0)
$writer.Write([uint16]1)
$writer.Write([uint16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $dim = if ($sizes[$i] -ge 256) { 0 } else { $sizes[$i] }
    $writer.Write([byte]$dim)
    $writer.Write([byte]$dim)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]$images[$i].Length)
    $writer.Write([uint32]$offset)
    $offset += $images[$i].Length
}
foreach ($image in $images) { $writer.Write($image) }
$writer.Close()

& $Csc /nologo /target:winexe /optimize+ "/win32icon:$IcoPath" "/out:$ExePath" `
    /reference:System.Windows.Forms.dll /reference:System.Drawing.dll (Join-Path $PSScriptRoot "Launcher.cs")
if ($LASTEXITCODE -ne 0) { throw "Compilation failed." }

$shortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Homelab AI.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $ExePath
$shortcut.WorkingDirectory = $Dist
$shortcut.IconLocation = $ExePath
$shortcut.Description = "Start Homelab AI"
$shortcut.Save()

Write-Host "Built $ExePath"
Write-Host "Desktop shortcut: $shortcutPath"
